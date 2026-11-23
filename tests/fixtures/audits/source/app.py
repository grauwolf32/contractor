from fastapi import Depends, FastAPI, HTTPException

app = FastAPI()


def authorize_widget(widget_id: str, actor: str) -> None:
    if actor != "fixture-owner":
        raise HTTPException(status_code=403, detail="forbidden")


@app.get("/widgets/{widget_id}")
def get_widget(widget_id: str, actor: str = Depends(str)) -> dict[str, str]:
    authorize_widget(widget_id, actor)
    return {"id": widget_id}


@app.delete("/widgets/{widget_id}", status_code=204)
def delete_widget(widget_id: str, actor: str = Depends(str)) -> None:
    authorize_widget(widget_id, actor)
