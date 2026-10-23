# Local Agent: MCP retrieval and durable human review

This local entry point connects `ReviewAgent` to the existing persistent Chroma
collection through a real MCP stdio client. Both `SEARCH` and `READ` invoke MCP;
READ no longer merely copies the search cache. The model cannot choose a server
command, path, URL, transport or extra tool. The fixed local subprocess exposes
only `search_clauses` and `read_evidence`.

This implementation extends `651b0a5e0b629a08541da74d8c4e6b49186e2e08` and is
published only to the existing draft PR8 after explicit authorization. Local
verification passed 48 targeted tests; exact-commit CI is reported separately.
No external client registration, listeners, credentials, model download,
deployment or merge is involved.

## Start and inspect

Use the existing project environment containing `requirements-agent.txt` and
`requirements-knowledge.txt`. Run from the repository root:

```powershell
python -m src.insurerag_vlm.agent_mcp_cli start --thread-id example-1 --question "What does the archived guidance say about prescription drug spending reporting?"
python -m src.insurerag_vlm.agent_mcp_cli status --thread-id example-1
```

Defaults are `.knowledge/dol-aca-20261001/db`, collection `dol_aca_20261001`, and
private checkpoint database `.agent-state/mcp-review.sqlite`. Optional `--store`,
`--collection` and `--db` select an existing local corpus and separate checkpoint.
Use the same configuration and `thread-id` for every subsequent command.

The default controller is **`mock-excerpt`**, a deterministic workflow driver:
it reads a retrieved clause and proposes a quoted historical-source excerpt.
It does not reason about whether that excerpt answers the question and must not
be reported as real model quality. `--controller mock-refusal` exercises a
refusal draft. The stored DOL source is real archived text, not a fictional
customer policy. Its 16 source documents / 303 chunks remain unchanged.

The output includes `result.approval`, with draft or refusal, exact evidence,
source IDs and hashes, dates/offsets where available, version, and approval ID.
`result.released_response` is **null until explicit approval**. Draft fields and
the `abstention` field are review material, not released responses. Neither a
timeout nor an absent/invalid decision can approve anything.

The new entry point reviews both ANSWER and ABSTAIN drafts. Operational failures
(invalid actions, missing evidence, tool failures or exhausted budgets) stop
without releasing a response; they are not silently converted into approved
refusals. Older Agent demo behavior is retained by the default
`review_abstentions=False` constructor option.

## Explicit review commands

Copy `version` and `approval_id` from the pending state into a local JSON file:

```json
{
  "decision_id": "review-1",
  "reviewer": "local-operator",
  "version": 1,
  "approval_id": "COPY_THE_PENDING_APPROVAL_ID",
  "action": "approve"
}
```

```powershell
python -m src.insurerag_vlm.agent_mcp_cli resume --thread-id example-1 --decision .agent-state/decision.json
```

| Action | Additional fields | Effect |
| --- | --- | --- |
| `approve` | None | Revalidate cited MCP evidence, then release this bound version |
| `edit` for an answer | `claims` using the existing claim/citation format | Validate exact quotes and current evidence, create a new version, pause again |
| `edit` for a refusal | `reason` (nonempty, at most 600 characters) | Change the refusal reason, create a new version, pause again |
| `retrieve` | New `query` and concrete `gap` | Clear prior MCP evidence/candidates/calculations, retrieve and read again within the remaining budgets |
| `reject` | None | End without release; MCP/model service is not required |

Use a new `decision_id` for each changed decision. Exact duplicate submissions
return the recorded state; conflicting reuse is rejected. Old versions and
other threads' approval IDs cannot release a revised draft. Human edits always
require a second explicit review. The reviewer field is a local operator label,
not authentication; this remains a trusted single-machine workflow.

## Evidence freshness, persistence and tool boundaries

Search results retain the Chroma evidence ID and full record-version SHA-256.
The graph's bounded evidence IDs map to these references. A READ sends both
values to MCP and checks the returned record hash, identity and exact text.
Unknown and stale evidence fail closed. Every approved claim still needs an
exact quotation from a successfully read record.

Before approval or editing, the current live MCP session revalidates stored
evidence versions. A changed/deleted record leaves the pending review intact
and rejects the decision; the operator can return it to retrieval or reject it.
A missing validating adapter cannot bypass this check. The stdio server holds
the collection's process lock for the Agent command, preventing imports between
validation and release. Stop the command before importing corpus changes.
The freshness cache lives only for that locked session, never across restarts.

LangGraph checkpoints, tool/controller result journals, approval bindings and
decisions remain in SQLite. Repeating `start` with the same question returns the
pending checkpoint without rerunning search/read/controller work. A different
question or controller/runtime configuration requires another thread. A complete
READ result survives a crash before its graph checkpoint; an uncertain operation
is not automatically repeated. A later approval still revalidates evidence in
its fresh MCP session. Those read-only freshness calls may repeat after a crash;
they do not repeat a generation or apply a decision twice.

`status` reads SQLite without connecting MCP or a model. `reject` also works
offline. Documents and questions remain untrusted data in the controller prompt.
Allowed actions and parameters are validated independently of the text; document
instructions cannot introduce shell/file/network tools or auto-approval.
Exact quotation validation does not establish entailment or current applicability;
those remain explicit review responsibilities.

## Existing local generation only

```powershell
python -m src.insurerag_vlm.agent_mcp_cli start --thread-id local-model-1 --question "YOUR QUESTION" --controller ollama --model ollama:EXISTING_MODEL --expected-digest EXISTING_LOCAL_DIGEST
```

Use the same model/controller flags on status/resume. The existing lazy local
controller only accesses `http://localhost:11434` and requires a supplied model
digest. It does not install/start/download a model or fall back to a paid API.
This run's local inventory probe was blocked with Windows socket error 10013;
service/model availability is therefore **unconfirmed**, not established absent.
No bypass or repeated network attempt was made. Actual-model evaluation needs
an allowed, reachable existing local service and its installed model identity.

Output always retains the scope warning: archived DOL guidance, current
applicability not revalidated, no personal policy determination, and
512-dimensional local-hashing rather than BGE. The older 384-dimensional vectors
are not used. Formal TEST/DEV and source acceptance states are unchanged.

## Local verification

```powershell
python -m pytest -q tests/test_agent_mcp.py tests/test_agent_graph.py
```

The tests use real stdio subprocesses and persistent Chroma/SQLite with fictional
fixtures and deterministic controllers. They cover approval before release,
restart and idempotence, edit/review/retrieval/rejection, reviewed refusal,
invented/unknown/stale/deleted evidence, model-requested forbidden actions,
MCP tool failure, and a completed READ followed by a checkpoint gap. Scripted
test approvals are not described as real human review.

Separately, the actual DOL collection is exercised by a mock-excerpt Agent run
which calls MCP search/read, verifies exact source offsets, and restarts into
the same pending approval. That smoke run deliberately submits **no approval**.
Its private session and receipt live under `.agent-state/mcp-integration-20261023/`.
This establishes real-corpus workflow wiring, not real-model answer quality.

## Page-aware evidence

Pass `--record-schema 2` with the isolated v2 store/collection for [page-aware ingestion](PAGE_INGEST_V2.md). Complete layout/context metadata survives MCP and the approval checkpoint. Generated figure summaries cannot serve as verbatim source quotations or numeric operands; context prefixes are retrieval aids only. Schema-1 pending approvals retain compatibility for unchanged evidence hashes.
