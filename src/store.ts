import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { MemoryFile, MemoryRecord } from "./types.js";

export interface MemoryStore {
  load(): Promise<MemoryRecord[]>;
  save(records: MemoryRecord[]): Promise<void>;
}

export class InMemoryStore implements MemoryStore {
  private records: MemoryRecord[] = [];
  async load(): Promise<MemoryRecord[]> {
    return structuredClone(this.records);
  }
  async save(records: MemoryRecord[]): Promise<void> {
    this.records = structuredClone(records);
  }
}

/** JSON file backend. Writes atomically (temp file + rename) so a crash never leaves a half-written file. */
export class JsonFileStore implements MemoryStore {
  constructor(readonly path: string) {}

  async load(): Promise<MemoryRecord[]> {
    let raw: string;
    try {
      raw = await readFile(this.path, "utf8");
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw err;
    }
    const data = JSON.parse(raw) as MemoryFile;
    if (data.version !== 1 || !Array.isArray(data.memories)) {
      throw new Error(`${this.path}: unsupported memory file format`);
    }
    return data.memories;
  }

  async save(records: MemoryRecord[]): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });
    const file: MemoryFile = { version: 1, memories: records };
    const tmp = `${this.path}.${process.pid}.tmp`;
    await writeFile(tmp, JSON.stringify(file, null, 2) + "\n", "utf8");
    await rename(tmp, this.path);
  }
}
