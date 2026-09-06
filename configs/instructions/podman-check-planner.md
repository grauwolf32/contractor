Delegate the Stage objective and exact source context to the single builder.
Require the builder's explicitly written `builder/check_report` ArtifactRef as
the `report` result. Local paths, command stdout and unpublished workspace files
are not Stage artifacts. Do not request workspace state or diff exports.
