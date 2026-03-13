import numpy as np
from PIL import Image
img = Image.open('_elev_debug/p2_region0_CONTENT_preview_tile.png')
gray = np.array(img.convert('L'), dtype=np.float32)
h, w = gray.shape
print(f'Image: {w}x{h}')

row_std = np.std(gray, axis=1)
col_std = np.std(gray, axis=0)

print(f'Row std first 20: {[round(v,1) for v in row_std[:20]]}')
print(f'Row std last 20: {[round(v,1) for v in row_std[-20:]]}')
print(f'Col std first 20: {[round(v,1) for v in col_std[:20]]}')
print(f'Col std last 20: {[round(v,1) for v in col_std[-20:]]}')

for i in range(h):
    if row_std[i] > 15:
        print(f'Row content starts at y={i} (std={row_std[i]:.1f})')
        break
for i in range(h-1, -1, -1):
    if row_std[i] > 15:
        print(f'Row content ends at y={i} (std={row_std[i]:.1f})')
        break
for j in range(w):
    if col_std[j] > 15:
        print(f'Col content starts at x={j} (std={col_std[j]:.1f})')
        break
for j in range(w-1, -1, -1):
    if col_std[j] > 15:
        print(f'Col content ends at x={j} (std={col_std[j]:.1f})')
        break

# Also check what the actual tile area looks like
# Look for strong edges using gradient
from scipy import ndimage
edges = ndimage.sobel(gray)
row_edge = np.mean(edges, axis=1)
col_edge = np.mean(edges, axis=0)
print(f'\nEdge mean first 20 rows: {[round(v,1) for v in row_edge[:20]]}')
print(f'Edge mean last 20 rows: {[round(v,1) for v in row_edge[-20:]]}')
