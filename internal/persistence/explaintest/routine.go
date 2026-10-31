package explaintest

// RoutineBodySQL reads a pg_proc row's SQL-standard or string body. alias is
// the trusted SQL alias of that row, or empty for an unqualified pg_proc read.
// A SQL-standard body has an empty prosrc; pg_get_function_sqlbody deparses
// its statements instead. Other languages keep their source in prosrc.
func RoutineBodySQL(alias string) string {
	if alias != "" {
		alias += "."
	}
	return "coalesce(pg_get_function_sqlbody(" + alias + "oid), " + alias + "prosrc)"
}
