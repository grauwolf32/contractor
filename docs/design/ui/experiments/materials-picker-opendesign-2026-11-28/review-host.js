(() => {
  let params = new URLSearchParams(location.search);
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
  window.addEventListener("message", (event) => {
    if (
      event.source !== frame.contentWindow ||
      event.origin !== location.origin ||
      event.data?.type !== "materials-preview-route"
    )
      return;
    const values = new URLSearchParams(event.data.query);
    if (
      ![...document.getElementById("screen").options].some(
        (option) => option.value === values.get("screen"),
      )
    )
      return;
    values.set("width", document.getElementById("width").value);
    for (const field of fields) {
      const value = values.get(field);
      if (
        [...document.getElementById(field).options].some(
          (option) => option.value === value,
        )
      )
        document.getElementById(field).value = value;
    }
    params = values;
    history.replaceState(null, "", `?${values}`);
    document.getElementById("direct").href = `prototype.html?${values}`;
    document.getElementById("state").disabled =
      values.get("screen") !== "materials";
  });
  function update() {
    const values = new URLSearchParams();
    for (const field of fields)
      values.set(field, document.getElementById(field).value);
    if (params.has("picker") && values.get("screen") === "check")
      values.set("picker", params.get("picker"));
    if (values.get("screen") !== "check") {
      for (const field of ["category", "search"])
        if (params.has(field)) values.set(field, params.get(field));
    }
    if (values.get("screen") === "material") {
      values.set("material", params.get("material") || "api-main");
      values.set("revision", params.get("revision") || "api-r4");
      values.set("state", "normal");
      document.getElementById("state").value = "normal";
    }
    params = values;
    history.replaceState(null, "", `?${values}`);
    frame.style.width = `${values.get("width")}px`;
    frame.src = `prototype.html?${values}`;
    document.getElementById("direct").href = frame.src;
    document.getElementById("state").disabled =
      values.get("screen") !== "materials";
    fitFrame();
  }
  update();
})();
