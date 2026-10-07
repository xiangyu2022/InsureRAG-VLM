"""Conservative acquisition access checks; never infer permission from bot identity."""
import math
import re
import time
from urllib.parse import unquote, urlparse


REVIEW_AGENTS = ('GPTBot', 'OAI-SearchBot', 'ChatGPT-User')


class ConservativeRobots:
    """Intersect every applicable group's restrictions, including wildcard.

    Allow exceptions intentionally do not override Disallow here. Acquisition may
    be narrower than a site's permission, but must not broaden that permission.
    """
    def __init__(self, text, agents):
        if re.search(r'<(?:html|!doctype)', text, re.I):
            raise ValueError('Ambiguous HTML robots response')
        groups = []
        names, rules = [], []
        for line in text.splitlines():
            line = line.split('#', 1)[0].strip()
            if ':' not in line:
                continue
            key, value = (part.strip() for part in line.split(':', 1))
            key = key.lower()
            if key == 'user-agent':
                if rules:
                    groups.append((names, rules))
                    names, rules = [], []
                names.append(value.lower())
            elif names:
                rules.append((key, value))
        groups.append((names, rules))
        identities = [agent.lower() for agent in agents]
        self.disallowed = []
        self.crawl_delay = 0.0
        for names, rules in groups:
            if not any(name == '*' or (name and any(name in agent for agent in identities)) for name in names):
                continue
            for key, value in rules:
                if key == 'disallow' and value:
                    if urlparse(value).scheme or value.startswith('//'):
                        parsed = urlparse(value)
                        value = (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
                    value = unquote(value)
                    end = value.endswith('$')
                    if end:
                        value = value[:-1]
                    self.disallowed.append(re.compile('^' + re.escape(value).replace(r'\*', '.*') + ('$' if end else '')))
                elif key == 'crawl-delay':
                    delay = float(value)
                    if not math.isfinite(delay) or delay < 0:
                        raise ValueError('Invalid robots crawl-delay')
                    self.crawl_delay = max(self.crawl_delay, delay)

    def can_fetch(self, url):
        parsed = urlparse(url)
        path = unquote(parsed.path or '/') + ('?' + unquote(parsed.query) if parsed.query else '')
        return not any(rule.search(path) for rule in self.disallowed)


def excluded_path(url, policy):
    path = unquote(urlparse(url).path).lower()
    return (any(path.startswith(prefix.lower()) for prefix in policy.get('excluded_prefixes', []))
            or any(word.lower() in path for word in policy.get('excluded_path_keywords', [])))


class AccessGuard:
    def __init__(self, robots, policy, content_check, *, clock=None, sleep=None):
        self.robots, self.policy, self.content_check = robots, policy, content_check
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        self.last_access = {}
        self.minimum = float(policy.get('minimum_interval_seconds', 1.2))
        if not math.isfinite(self.minimum) or self.minimum < 0:
            raise ValueError('Invalid minimum request interval')

    def __call__(self, url):
        parsed = urlparse(url)
        if (parsed.scheme != 'https' or parsed.username or parsed.password
                or parsed.port not in (None, 443) or parsed.hostname not in self.robots):
            raise ValueError('URL outside reviewed HTTPS host scope')
        if any(part in ('.', '..') for part in unquote(parsed.path).split('/')):
            raise ValueError('Ambiguous dot-segment URL requires review')
        parser = self.robots[parsed.hostname]
        if not parser.can_fetch(url):
            raise ValueError('robots_disallow')
        if excluded_path(url, self.policy) or not self.content_check(url):
            raise ValueError('content_scope_disallow')
        delay = max(self.minimum, parser.crawl_delay)
        previous = self.last_access.get(parsed.hostname)
        if previous is not None:
            self.sleep(max(0, previous + delay - self.clock()))
        self.last_access[parsed.hostname] = self.clock()
