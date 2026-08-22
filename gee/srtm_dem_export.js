// ============================================================
// GEE Export Script: SRTM Elevation (DEM) for Topographic Wetness
// Index (TWI) computation
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. Check console for errors before exporting
// 5. Go to "Tasks" tab, run the export task
// 6. File appears in Google Drive under "GEE_exports" once done
// ============================================================

var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);
Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

// SRTM 30m DEM - single static image, NOT a time series (elevation
// doesn't change month to month, unlike everything else you've used)
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region);

Export.image.toDrive({
  image: dem,
  description: 'SRTM_DEM_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'srtm_dem_amazon_cerrado',
  region: region,
  scale: 1000, // deliberately coarser than SRTM's native 30m - a full
               // flow-accumulation computation at native resolution
               // over this large a region would be both a huge file
               // and a slow computation, for little extra benefit once
               // averaged into ~90km patches anyway. 1km keeps this
               // both fast to compute and small to download.
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - Single-band, single-image file (not a monthly stack) - elevation
//   is static.
// - Flow accumulation and TWI itself will be CALCULATED in Python
//   from this elevation data (GEE has no built-in flow accumulation
//   function) - see the processing script for the actual algorithm.
// ============================================================