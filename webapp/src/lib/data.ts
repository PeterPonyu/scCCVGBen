import fs from 'fs/promises';
import path from 'path';
import type { Dataset, Methods, Metrics, Summary } from './types';

const DATA_DIR = path.join(process.cwd(), '..', 'site', 'data');

async function readJson<T>(filename: string): Promise<T> {
  const filePath = path.join(DATA_DIR, filename);
  const raw = await fs.readFile(filePath, 'utf-8');
  return JSON.parse(raw) as T;
}

export async function getDatasets(): Promise<Dataset[]> {
  return readJson<Dataset[]>('datasets.json');
}

export async function getMethods(): Promise<Methods> {
  return readJson<Methods>('methods.json');
}

export async function getMetrics(): Promise<Metrics> {
  return readJson<Metrics>('metrics.json');
}

export async function getSummary(): Promise<Summary> {
  return readJson<Summary>('summary.json');
}
