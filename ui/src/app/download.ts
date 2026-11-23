/**
 * Saves a blob as a file through a temporary download link. The object URL
 * is revoked after the click has been dispatched.
 */
export function saveBlob(blob: Blob, filename: string): void {
  const objectURL = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectURL;
  anchor.download = filename;
  anchor.hidden = true;
  document.body.append(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(objectURL), 0);
  }
}
