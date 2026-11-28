(() => {
  const params = new URLSearchParams(location.search);
  const frame = document.getElementById("preview");
  const fields = ["screen", "theme", "width", "state"];
  for (const field of fields) {
    const select = document.getElementById(field);
    const selected = params.get(field);
    if ([...select.options].some((option) => option.value === selected))
      select.value = selected;
    select.addEventListener("change", update);
  }
  function fitFrame() {
    const width = Number(document.getElementById("width").value);
    const scale = Math.min(1, Math.max(300, innerWidth - 24) / width);
    const height = Math.max(
      580,
      innerHeight - document.querySelector(".review-toolbar").offsetHeight - 16,
    );
    frame.style.transform = `scale(${scale})`;
    frame.style.height = `${height / scale}px`;
    document.querySelector(".review-stage").style.height = `${height}px`;
  }
  window.addEventListener("resize", fitFrame);
  function update() {
    const values = new URLSearchParams();
    for (const field of fields)
      values.set(field, document.getElementById(field).value);
    if (params.has("picker") && values.get("screen") === "check")
      values.set("picker", params.get("picker"));
    history.replaceState(null, "", `?${values}`);
    frame.style.width = `${values.get("width")}px`;
    frame.src = `prototype.html?${values}`;
    document.getElementById("direct").href = frame.src;
    document.getElementById("state").disabled =
      values.get("screen") === "check";
    fitFrame();
  }
  update();
})();
