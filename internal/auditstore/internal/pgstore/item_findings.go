package pgstore

import (
 "context"
 "github.com/grauwolf32/contractor/internal/auditstore"
 "github.com/grauwolf32/contractor/internal/helpers/pagelimit"
)

// ItemFindingIDs reads only a selected page's ordinary source associations.
// It neither chooses a result as current nor interprets producer subject text.
func (s *QueryStore) ItemFindingIDs(ctx context.Context,ownerID,auditID string,itemIDs []string) (map[string][]string,error) {
 result:=make(map[string][]string,len(itemIDs))
 if len(itemIDs)==0 { return result,nil }
 if len(itemIDs)>pagelimit.MaxRows+1 { return nil,auditstore.ErrInvalid }
 for _,id:=range itemIDs { result[id]=[]string{} }
 rows,err:=s.db.Query(ctx,itemFindingIDsSQL,ownerID,auditID,itemIDs)
 if err!=nil { return nil,err }
 defer rows.Close()
 for rows.Next() {
 var itemID,findingID string
 if err:=rows.Scan(&itemID,&findingID);err!=nil { return nil,err }
 result[itemID]=append(result[itemID],findingID)
 }
 return result,rows.Err()
}
