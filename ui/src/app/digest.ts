/** Lowercase hexadecimal SHA-256 of the given bytes. */
export async function sha256Hex(data: BufferSource): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hash), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}
