package projectworkflows

import (
	"archive/zip"
	"bytes"
	"sort"
	"time"
)

const ApplicationSource = `import os

import asyncpg
import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

app = FastAPI(title="Widget Service")
bearer = HTTPBearer()


class WidgetCreate(BaseModel):
    name: str


def require_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
) -> None:
    if credentials.credentials != os.environ["API_TOKEN"]:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/widgets/{widget_id}", dependencies=[Depends(require_token)])
async def get_widget(widget_id: str) -> dict[str, object]:
    connection = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        row = await connection.fetchrow(
            "SELECT id, name FROM widgets WHERE id = $1", widget_id
        )
    finally:
        await connection.close()
    if row is None:
        raise HTTPException(status_code=404, detail="widget not found")
    async with httpx.AsyncClient(base_url=os.environ["INVENTORY_URL"]) as client:
        inventory = await client.get(f"/v1/items/{widget_id}")
    return {
        "id": row["id"],
        "name": row["name"],
        "available": inventory.status_code == 200,
    }


@app.post(
    "/widgets",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_token)],
)
async def create_widget(widget: WidgetCreate) -> dict[str, str]:
    connection = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        widget_id = await connection.fetchval(
            "INSERT INTO widgets(name) VALUES($1) RETURNING id", widget.name
        )
    finally:
        await connection.close()
    return {"id": str(widget_id), "name": widget.name}
`

const ProjectManifest = `[project]
name = "widget-service"
version = "1.0.0"
dependencies = [
  "asyncpg>=0.30",
  "fastapi>=0.116",
  "httpx>=0.28",
  "pydantic>=2.11",
]
`

var ExpectedRoutes = map[string][]string{
	"/widgets":             {"post"},
	"/widgets/{widget_id}": {"get"},
}

func SourceArchive() ([]byte, error) {
	files := map[string]string{
		"app.py":         ApplicationSource,
		"pyproject.toml": ProjectManifest,
		"README.md":      "# Widget Service\n\nAuthenticated API backed by PostgreSQL and Inventory.\n",
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	for _, name := range names {
		header := &zip.FileHeader{Name: name, Method: zip.Store}
		header.SetMode(0o600)
		header.SetModTime(time.Date(2026, 8, 30, 0, 0, 0, 0, time.UTC))
		entry, err := writer.CreateHeader(header)
		if err != nil {
			return nil, err
		}
		if _, err := entry.Write([]byte(files[name])); err != nil {
			return nil, err
		}
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}
