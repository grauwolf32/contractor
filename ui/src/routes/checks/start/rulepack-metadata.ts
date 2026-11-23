const YAML_LIMIT = 1024 * 1024;
const ZIP_LIMIT = 16 * 1024 * 1024;

export const NUCLEI_RULEPACK_ZIP =
  "application/vnd.contractor.nuclei-rulepack+zip";

/** SCN-R239 metadata admission; empty content is validated separately. */
export function rulepackMetadataError(
  mediaType: string,
  size: number,
): string | undefined {
  if (!Number.isSafeInteger(size) || size < 0)
    return "The rulepack size is invalid.";
  if (mediaType !== "application/yaml" && mediaType !== NUCLEI_RULEPACK_ZIP)
    return "Choose a Nuclei YAML or ZIP rulepack.";
  const limit = mediaType === "application/yaml" ? YAML_LIMIT : ZIP_LIMIT;
  if (size > limit)
    return `Choose a ${mediaType === "application/yaml" ? "YAML rule" : "ZIP rulepack"} of at most ${limit / (1024 * 1024)} MiB.`;
  return undefined;
}
