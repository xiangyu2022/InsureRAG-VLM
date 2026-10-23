"""Version 2 page/block provenance. Unknown layout is explicit, never invented."""
from __future__ import annotations

import math
from typing import Generic, Literal, TypeVar
from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar('T')
SHA = str


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Observed(StrictModel, Generic[T]):
    status: Literal['observed', 'not_found', 'unreadable', 'not_applicable', 'unknown_legacy']
    value: T | None
    extractor: str = Field(min_length=1)
    confidence: float | None = Field(ge=0, le=1)
    needs_review: bool

    @model_validator(mode='after')
    def check_value(self):
        if (self.status == 'observed') != (self.value is not None):
            raise ValueError('Only observed fields may have a value')
        return self


class Box(StrictModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode='after')
    def check_box(self):
        if not all(math.isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError('Non-finite coordinates')
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise ValueError('Box must be normalized to the original page')
        return self


class Segment(StrictModel):
    kind: Literal['paragraph', 'clause', 'heading', 'header', 'footer', 'caption']
    text: str = Field(min_length=1)
    bbox: Box | None
    heading_level: Observed[int]

    @model_validator(mode='after')
    def check_heading_level(self):
        if self.heading_level.value is not None and not 1 <= self.heading_level.value <= 6:
            raise ValueError('Heading level must be 1..6 or explicitly unknown')
        return self


class Cell(StrictModel):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(ge=1)
    column_span: int = Field(ge=1)
    text: Observed[str]
    bbox: Box


class Table(StrictModel):
    table_id: str | None = None
    bbox: Box
    number: Observed[str]
    caption: Observed[str]
    header_rows: Observed[list[int]]
    cells: list[Cell] = Field(min_length=1)
    units: Observed[list[str]]
    footnotes: Observed[list[str]]
    continued: Observed[bool]

    @model_validator(mode='after')
    def check_cells(self):
        occupied = set()
        for cell in self.cells:
            if not (self.bbox.x0 <= cell.bbox.x0 < cell.bbox.x1 <= self.bbox.x1 and
                    self.bbox.y0 <= cell.bbox.y0 < cell.bbox.y1 <= self.bbox.y1):
                raise ValueError('Cell lies outside its table')
            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(cell.column, cell.column + cell.column_span):
                    if (row, column) in occupied:
                        raise ValueError('Overlapping table cells')
                    occupied.add((row, column))
        if self.header_rows.value is not None and any(r < 0 or not any(c.row <= r < c.row + c.row_span for c in self.cells) for r in self.header_rows.value):
            raise ValueError('Header row is not in the table')
        return self


class Figure(StrictModel):
    figure_id: str | None = None
    kind: Literal['image', 'chart']
    bbox: Box
    caption: Observed[str]
    summary: Observed[str]
    axes: Observed[list[str]]
    legend: Observed[list[str]]
    units: Observed[list[str]]
    numeric_uncertainty: Observed[list[str]]


class PageExtraction(StrictModel):
    readability: Literal['readable', 'unreadable', 'uncertain']
    printed_page: Observed[str]
    segments: list[Segment]
    tables: Observed[list[Table]]
    figures: Observed[list[Figure]]


class Location(StrictModel):
    page_id: str = Field(min_length=1)
    physical_page: int = Field(ge=1)
    bbox: Box | None
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    coordinate_basis: Literal['normalized_original_page', 'logical_text_offsets']

    @model_validator(mode='after')
    def check_range(self):
        if self.char_end <= self.char_start:
            raise ValueError('Empty/reversed character range')
        return self


class Block(StrictModel):
    block_id: str
    kind: str
    order: int = Field(ge=1)
    parent_id: str | None
    location: Location
    text: str
    heading_level: Observed[int]


class Section(StrictModel):
    heading_id: str
    title: str
    level: Observed[int]
    source_page_id: str
    inherited: bool
    source_location: Location


class ImageProcessing(StrictModel):
    asset: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    format: Literal['JPEG']
    dpi: int = Field(gt=0)
    quality: int = Field(ge=75, le=95)
    max_edge: int = Field(ge=800, le=3000)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    original_width: float = Field(gt=0)
    original_height: float = Field(gt=0)
    resize_filter: str
    converter_version: str
    min_native_font_pixels: float | None
    readability_status: Literal['not_checked', 'backend_declared_readable', 'uncertain']


class Processing(StrictModel):
    route: Literal['native_text', 'compressed_multimodal']
    parser_version: str = Field(min_length=1)
    model_version: str | None
    image: ImageProcessing | None
    extraction_reviewed: bool

    @model_validator(mode='after')
    def check_route(self):
        if self.route == 'compressed_multimodal' and (not self.image or not self.model_version):
            raise ValueError('Visual processing requires image provenance and model version')
        if self.route == 'native_text' and (self.image is not None or self.model_version is not None):
            raise ValueError('Native text must not claim visual processing')
        return self


class PageMetadata(StrictModel):
    page_id: str
    page_number: int = Field(ge=1)
    physical_page: Observed[int]
    printed_page: Observed[str]
    headers: Observed[list[Block]]
    footers: Observed[list[Block]]
    headings: Observed[list[Block]]
    sections: Observed[list[Section]]
    captions: Observed[list[Block]]
    paragraphs: Observed[list[Block]]
    tables: Observed[list[Table]]
    figures: Observed[list[Figure]]
    blocks: list[Block]

    @model_validator(mode='after')
    def check_blocks(self):
        seen = set()
        for i, block in enumerate(self.blocks, 1):
            if block.order != i or block.block_id in seen or (block.parent_id and block.parent_id not in seen):
                raise ValueError('Invalid block order, parent or stable ID')
            if block.location.page_id != self.page_id or block.location.physical_page != self.page_number:
                raise ValueError('Block crosses page boundary')
            seen.add(block.block_id)
        for section in self.sections.value or []:
            if section.inherited != (section.source_page_id != self.page_id):
                raise ValueError('Inherited section must name its original page')
            if not section.inherited:
                heading = next((b for b in self.blocks if b.block_id == section.heading_id), None)
                if not heading or heading.text != section.title or heading.location != section.source_location:
                    raise ValueError('Section context is not bound to an actual heading block')
        return self


class ChunkMetadata(StrictModel):
    chunk_id: str
    sequence: int = Field(ge=1)
    unit_id: str
    block_ids: list[str] = Field(min_length=1)
    locations: list[Location] = Field(min_length=1)
    content_type: Literal['paragraph', 'clause', 'heading', 'header', 'footer', 'caption', 'table_rows', 'figure_summary']
    evidence_origin: Literal['native_text', 'multimodal_transcription', 'generated_summary']
    semantic_boundary: str
    continuation_index: int = Field(ge=1)
    oversized_unsplit: bool
    table_id: str | None
    table: Table | None
    row_indices: list[int] | None
    cross_page_table_group: Observed[str]
    figure_id: str | None
    figure: Figure | None


def validate_page_record(row):
    if row.get('schema_version') != 2:
        raise ValueError('Page records require schema_version=2')
    required = {'record_id', 'doc_id', 'source_sha256', 'source_file', 'page_number', 'page_id',
                'chunk_id', 'chunk_sequence', 'processing', 'page_metadata', 'chunk_metadata', 'content_type', 'evidence_origin',
                'source_text', 'summary_text', 'context_prefix', 'context_metadata', 'retrieval_text'}
    if not required <= row.keys():
        raise ValueError('Missing mandatory page/chunk provenance')
    if not row['source_file'] or not row['doc_id']:
        raise ValueError('Missing source identity')
    page = PageMetadata.model_validate(row['page_metadata'])
    chunk = ChunkMetadata.model_validate(row['chunk_metadata'])
    Processing.model_validate(row['processing'])
    if page.page_number != row['page_number'] or page.page_id != row['page_id']:
        raise ValueError('Page identity mismatch')
    if chunk.chunk_id != row['record_id'] or chunk.chunk_id != row['chunk_id'] or chunk.sequence != row['chunk_sequence']:
        raise ValueError('Chunk identity mismatch')
    if chunk.content_type != row['content_type'] or chunk.evidence_origin != row['evidence_origin']:
        raise ValueError('Evidence type mismatch')
    block_ids = {b.block_id for b in page.blocks}
    if not set(chunk.block_ids) <= block_ids:
        raise ValueError('Unknown chunk block')
    for location in chunk.locations:
        if location.page_id != page.page_id or location.physical_page != page.page_number:
            raise ValueError('Chunk spans physical pages')
    if chunk.content_type == 'figure_summary' and (not chunk.figure_id or not chunk.figure or chunk.evidence_origin != 'generated_summary'):
        raise ValueError('Figure summary must retain original figure and generated status')
    if chunk.content_type == 'table_rows' and (not chunk.table_id or not chunk.table or chunk.row_indices is None):
        raise ValueError('Table chunks must retain table/header/row structure')
    if chunk.table is not None and chunk.table.table_id != chunk.table_id:
        raise ValueError('Table identity mismatch')
    if chunk.figure is not None and chunk.figure.figure_id != chunk.figure_id:
        raise ValueError('Figure identity mismatch')
    if row['evidence_origin'] == 'generated_summary':
        if row['source_text'] is not None or row['summary_text'] != row['text']:
            raise ValueError('Generated summary cannot masquerade as source text')
    elif row['source_text'] != row['text'] or row['summary_text'] is not None:
        raise ValueError('Source text must stay separate from generated context')
    from .page_ingest import context_prefix
    prefix, context = context_prefix(row)
    if row['context_prefix'] != prefix or row['context_metadata'] != context:
        raise ValueError('Context prefix is stale or not grounded in the stored structure')
    if row['retrieval_text'] != prefix + '\n\n' + row['text']:
        raise ValueError('Retrieval text does not match prefix and source version')
