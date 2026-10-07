const CACHE_NAME = "kalkulator-metadata-v1";

/** Small, immutable model files. Weight shards stay in their existing OPFS cache. */
export async function loadModelMetadata(
  url: string,
  sha256: string,
  label: string,
): Promise<Uint8Array<ArrayBuffer>> {
  let cache: Cache | undefined;
  try {
    cache = await caches.open(CACHE_NAME);
    const response = await cache.match(url);
    if (response) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (await matchesHash(bytes, sha256)) return bytes;
      await cache.delete(url);
      console.warn(`Replacing damaged cached ${label}.`);
    }
  } catch (error) {
    console.warn(`Could not read cached ${label}.`, error);
  }

  let response: Response;
  try {
    response = await fetch(url, { cache: "no-cache", signal: AbortSignal.timeout(60_000) });
  } catch (cause) {
    throw new Error(`Could not download ${label}. Check your connection and reload.`, { cause });
  }
  if (!response.ok) throw new Error(`Could not download ${label} (${response.status}).`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (!(await matchesHash(bytes, sha256))) {
    throw new Error(`The downloaded ${label} is damaged. Reload to try again.`);
  }
  try {
    await cache?.put(url, new Response(bytes, { headers: { "Content-Type": "application/json" } }));
  } catch (error) {
    console.warn(`Could not keep ${label} on this device.`, error);
  }
  return bytes;
}

async function matchesHash(bytes: Uint8Array<ArrayBuffer>, expected: string): Promise<boolean> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("") === expected;
}
