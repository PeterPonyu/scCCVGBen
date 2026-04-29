// Dataset record from site/data/datasets.json
export interface Dataset {
  id: string;
  GSE: string;
  modality: string;
  species: string;
  tissue: string;
  cell_count: number;
  category: string;
  description: string;
  geo_title: string;
  geo_url: string;
  pubmed_id: string;
  submission_date: string;
  source_name: string;
  cell_count_status?: string;
  release_status?: 'public' | 'restricted';
}

// Method record from site/data/methods.json
export interface Method {
  name: string;
  family: string;
  description: string;
}

export interface Methods {
  scCCVGBen_encoders: Method[];
  graph_constructions: Method[];
  baselines: Method[];
}

// Metric record from site/data/metrics.json
export interface Metric {
  name: string;
  description: string;
  source: string;
  note?: string;
}

export interface Metrics {
  clustering: Metric[];
  dre: Metric[];
  lse: Metric[];
}

// Summary from site/data/summary.json
export interface CellCountBin {
  bin: string;
  count: number;
}

export interface YearCount {
  year: number;
  count: number;
}

export interface TissueModality {
  tissue: string;
  modality: string;
  count: number;
}

export interface SpeciesModality {
  species: string;
  modality: string;
  count: number;
}

export interface PubmedTop {
  id: string;
  GSE: string;
  pubmed_id: string;
  cell_count: number;
  species: string;
  modality: string;
}

export interface Summary {
  built_at: string;
  total_datasets: number;
  restricted_datasets?: number;
  modality: Record<string, number>;
  species: Record<string, number>;
  tissue_top10: Record<string, number>;
  species_modality: SpeciesModality[];
  tissue_modality: TissueModality[];
  cell_count_range: { min: number; median: number; max: number };
  cell_count_missing?: number;
  cell_count_hist: CellCountBin[];
  submission_year_timeline: YearCount[];
  pubmed_top15: PubmedTop[];
  methods_total: number;
  metrics_total: number;
}
