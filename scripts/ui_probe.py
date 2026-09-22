"""Headless UI check: drives the viewer in Chromium (SwiftShader, slow) and prints view state.

Start the dev server first:  npx vite --port 5199 --strictPort
Then:                         uv run --with playwright python scripts/ui_probe.py
Screenshots go to $PROBE_OUT (default: current directory). Waits on window.flywire.state, not on time:
software rendering is slow, so a 1.4 s transition takes much longer here.
"""
import os
import asyncio
from playwright.async_api import async_playwright

D = os.environ.get("PROBE_OUT", ".").rstrip("/") + "/"
STATE = "() => flywire.state"
SETTLED = "() => flywire.state.shown === flywire.state.layout && flywire.state.blend >= 1"


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = await b.new_page(viewport={"width": 1400, "height": 850})
        errors = []
        pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
        pg.on("pageerror", lambda e: errors.append(str(e)))
        await pg.goto("http://localhost:5199/")
        await pg.wait_for_selector("#loading", state="detached", timeout=120000)
        await pg.wait_for_timeout(1500)
        print("anatomical:", await pg.evaluate(STATE))
        await pg.screenshot(path=D + "1_anatomical.png")

        for n, lid in enumerate(["soma", "mirrored", "flat", "partners", "flow"], start=2):
            await pg.click(f'[data-layout="{lid}"]')
            await pg.wait_for_function(f"() => flywire.state.shown === '{lid}' && flywire.state.blend >= 1", timeout=300000)
            await pg.wait_for_timeout(2500)
            print(f"{lid}:", await pg.evaluate(STATE))
            await pg.screenshot(path=D + f"{n}_{lid}.png")

        # Colour by + search + selection with partners.
        await pg.select_option("#colour-by", "flow")
        await pg.fill("#focus-search", "Kenyon")
        await pg.wait_for_selector("#focus-results li")
        print("search hits:", await pg.eval_on_selector_all("#focus-results li .r-name", "els => els.slice(0,5).map(e => e.textContent)"))
        await pg.keyboard.press("Enter")
        await pg.wait_for_timeout(2000)
        print("focus:", await pg.evaluate(STATE))
        await pg.click('[data-layout="anatomical"]')
        await pg.wait_for_function(SETTLED, timeout=300000)
        await pg.evaluate("() => flywire.select(flywire.groupings.get('cell_type').codes.findIndex(c => flywire.groupings.get('cell_type').values[c] === 'DA1_lPN'))")
        await pg.wait_for_function("() => flywire.state.partners > 0", timeout=120000)
        await pg.wait_for_timeout(2000)
        print("selected:", await pg.evaluate(STATE))
        print("card:", (await pg.inner_text("#info"))[:600].replace("\n", " | "))
        await pg.screenshot(path=D + "11_partners.png")

        # Wiring groupings: colour by each one and read the legend.
        await pg.keyboard.press("Escape")
        await pg.click('[data-layout="partners"]')
        await pg.wait_for_function(SETTLED, timeout=300000)
        for n, gid in enumerate(["leiden_coarse", "leiden_fine", "infomap", "conn_type", "conn_kmeans", "hub_band"], start=12):
            await pg.select_option("#colour-by", gid)
            await pg.wait_for_timeout(1500)
            rows = await pg.eval_on_selector_all("#legend .legend-row", "els => els.map(e => e.innerText.replace(/\\n/g, ' '))")
            print(f"legend {gid}:", rows[:3], "…", rows[-2:])
            await pg.screenshot(path=D + f"{n}_{gid}.png")
        await pg.fill("#focus-search", "T4a")
        await pg.wait_for_selector("#focus-results li")
        print("search T4a:", await pg.eval_on_selector_all("#focus-results li", "els => els.slice(0,6).map(e => e.innerText.replace(/\\n/g, ' '))"))
        await pg.fill("#focus-search", "")
        await pg.keyboard.press("Escape")

        # Connection matrix for region, in the anatomical layout.
        await pg.click('[data-layout="anatomical"]')
        await pg.wait_for_function(SETTLED, timeout=300000)
        await pg.select_option("#colour-by", "region")
        await pg.click("#matrix-toggle")
        await pg.wait_for_selector("#matrix table.matrix", timeout=120000)
        print("matrix size:", await pg.eval_on_selector("#matrix tbody", "t => [t.rows.length, t.rows[0].cells.length]"))
        print("matrix diagonal:", await pg.eval_on_selector_all("#matrix tbody tr", "rs => rs.slice(0,3).map((r, k) => r.cells[k + 1].getAttribute('aria-label'))"))
        await pg.hover("#matrix tbody tr:nth-child(1) td:nth-child(5)")
        await pg.wait_for_timeout(1500)
        print("matrix hover:", await pg.inner_text(".matrix-readout"))
        await pg.screenshot(path=D + "18_matrix_share.png")
        await pg.click('[data-scale="synapses"]')
        await pg.mouse.move(700, 100)
        await pg.wait_for_timeout(1500)
        await pg.screenshot(path=D + "19_matrix_synapses.png")
        await pg.select_option("#colour-by", "cell_type")
        await pg.wait_for_timeout(1500)
        print("matrix cell_type rows:", await pg.eval_on_selector_all("#matrix tbody th", "els => els.map(e => e.innerText)"))
        await pg.screenshot(path=D + "20_matrix_cell_type.png")
        print("console errors:", errors or "none")
        await b.close()

asyncio.run(main())
