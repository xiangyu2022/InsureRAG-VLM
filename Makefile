PYTHON  ?= python
DATA    ?= data/00_raw/external/public_docs
INDEX   ?= data
VISUAL  ?= data/03_index/colqwen2
QA      ?= data/02_processed/qa_pairs.jsonl
REPORTS ?= reports

.PHONY: help local-setup smoke-test install index visual-index eval ablation clean

help:
	@echo "InsureRAG-VLM — available targets"
	@echo "  make local-setup  Create .venv and install local dependencies"
	@echo "  make install       Install Python dependencies"
	@echo "  make smoke-test    End-to-end no-key smoke test on real public PDFs"
	@echo "  make index         Build text index from public docs"
	@echo "  make visual-index  Build visual_stub + local_image indexes"
	@echo "  make eval          Run retrieval metrics on QA set"
	@echo "  make ablation      Run full ablation (all CPU backends)"
	@echo "  make clean         Remove generated indexes and reports"

install:
	pip install -r requirements.txt

local-setup:
	bash scripts/setup_local.sh

smoke-test:
	PYTHON=$(PYTHON) bash scripts/smoke_test.sh

index:
	$(PYTHON) main.py build-index $(DATA)

visual-index:
	$(PYTHON) main.py preprocess-pages $(DATA) --output-root data
	$(PYTHON) main.py build-visual-index $(VISUAL)/page_manifest.jsonl --backend visual_stub
	$(PYTHON) main.py build-visual-index $(VISUAL)/page_manifest.jsonl --backend local_image

eval:
	$(PYTHON) main.py retrieval-metrics $(DATA) $(QA)
	$(PYTHON) main.py visual-retrieval-metrics $(QA) --backend visual_stub
	$(PYTHON) main.py visual-retrieval-metrics $(QA) --backend local_image

ablation:
	mkdir -p $(REPORTS)/ablation_real_pdfs
	$(PYTHON) main.py run-ablation \
		--data-folder $(DATA) \
		--qa-path $(QA) \
		--output-dir $(REPORTS)/ablation_real_pdfs \
		--visual-index-dir $(VISUAL)

clean:
	rm -rf data/index.npy data/index_meta.json
	rm -rf $(VISUAL)/visual_stub.npy $(VISUAL)/local_image.npy
	rm -rf $(VISUAL)/visual_stub_pages.jsonl $(VISUAL)/local_image_pages.jsonl
