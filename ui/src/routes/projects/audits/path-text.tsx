import { Fragment } from "react";

/**
 * An endpoint path that may wrap after each "/", so long paths break between
 * their parts instead of anywhere. The text (and accessible name) is the
 * path as written.
 */
export function PathText({ path }: { path: string }) {
  const parts = path.split("/");
  return (
    <>
      {parts.map((part, index) => (
        // Path parts are positional.
        <Fragment key={index}>
          {part}
          {index < parts.length - 1 ? (
            <>
              /<wbr />
            </>
          ) : null}
        </Fragment>
      ))}
    </>
  );
}
