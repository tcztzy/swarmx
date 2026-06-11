export class QuotaManager {
  private used = 0;
  private limit: number;
  private lock = false;
  private waitQueue: Array<() => void> = [];

  constructor(limit: number) {
    this.limit = limit;
  }

  get remaining(): number {
    return Math.max(0, this.limit - this.used);
  }

  get usage(): number {
    return this.used;
  }

  private async acquire(): Promise<void> {
    if (!this.lock) {
      this.lock = true;
      return;
    }
    return new Promise((resolve) => {
      this.waitQueue.push(resolve);
    });
  }

  private release(): void {
    const next = this.waitQueue.shift();
    if (next) {
      next();
    } else {
      this.lock = false;
    }
  }

  async consume(tokens: number): Promise<void> {
    await this.acquire();
    try {
      this.used += tokens;
      if (this.used > this.limit) {
        console.warn(`Quota exceeded: used ${this.used}/${this.limit} tokens`);
      }
    } finally {
      this.release();
    }
  }

  async checkAndConsume(tokens: number): Promise<void> {
    await this.acquire();
    try {
      if (this.used + tokens > this.limit) {
        throw new Error(`Token quota exceeded: ${this.used + tokens}/${this.limit}`);
      }
      this.used += tokens;
    } finally {
      this.release();
    }
  }

  reset(): void {
    this.used = 0;
  }

  setLimit(limit: number): void {
    this.limit = limit;
  }
}
