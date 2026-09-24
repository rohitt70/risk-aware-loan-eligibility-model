"""patch_charts.py — injects _chart_layout() before every st.plotly_chart call"""
import re

src = open("app.py", encoding="utf-8").read()
lines = src.split("\n")
new_lines = []

for line in lines:
    stripped = line.lstrip()
    indent   = " " * (len(line) - len(stripped))
    # Before any st.plotly_chart(varname, ...) that isn't from perf_gauge
    if "st.plotly_chart(" in stripped and "perf_gauge" not in stripped:
        m = re.match(r"st\.plotly_chart\((\w+),", stripped)
        if m:
            varname = m.group(1)
            new_lines.append(f"{indent}{varname}.update_layout(**_chart_layout())")
    new_lines.append(line)

result = "\n".join(new_lines)
open("app.py", "w", encoding="utf-8").write(result)
n = result.count("_chart_layout()")
print(f"Done — {n} _chart_layout() calls in app.py")
