# Synthetic Policy Packet Examples

The directory `examples/policy_packets/` contains synthetic, redistribution-safe policy packets for testing packet-aware hybrid RAG behavior.

They are not carrier forms and should not be treated as legal policy language. Their purpose is to exercise:

- declarations-page limit and deductible extraction
- base-policy coverage and exclusion retrieval
- endorsement override / add-back reasoning
- schedule lookup for item-specific limits
- cross-document graph expansion within a packet

Use them with:

```bash
.venv_py313/bin/python main.py build-index examples/policy_packets --index-dir /tmp/insurerag_policy_packet_index --retrieval-mode hybrid_text --corpus-source documents --disable-image-signal
.venv_py313/bin/python main.py query examples/policy_packets "Does the water backup endorsement change the water exclusion?" --index-dir /tmp/insurerag_policy_packet_index --retrieval-mode hybrid_text --corpus-source documents --disable-image-signal --json
```

