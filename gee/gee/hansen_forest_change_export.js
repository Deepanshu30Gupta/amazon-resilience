// ============================================================
// GEE Export Script: Hansen Global Forest Change
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. IMPORTANT: check the console for errors - I could not verify this
//    live. Search the GEE Data Catalog for "Hansen Global Forest
//    Change" if 'UMD/hansen/global_forest_change_2023_v1_11' has been
//    superseded by a newer version - just swap in the current ID,
//    the band names ('treecover2000', 'lossyear') have stayed stable
//    across versions for years.
// 5. Go to "Tasks" tab, run the export task
// 6. File appears in Google Drive under "GEE_exports" once done
// ============================================================

var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);
Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

var hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11');

// treecover2000: % tree cover in year 2000 (0-100)
// lossyear: year forest loss was detected (0 = no loss, 1-23 = 2001-2023)
// We only care about loss THROUGH our study period (2003-2018), i.e. lossyear 1-18
var treecover = hansen.select('treecover2000').clip(region);
var lossyear = hansen.select('lossyear').clip(region);

// Combine into a 2-band image: band 0 = tree cover 2000, band 1 = loss year
var combined = treecover.rename('treecover2000').addBands(lossyear.rename('lossyear'));

Export.image.toDrive({
  image: combined,
  description: 'Hansen_forest_change_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'hansen_forest_change_amazon_cerrado',
  region: region,
  scale: 100, // aggregate from Hansen's native 30m to 100m to keep file size manageable
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - This is a SINGLE static image (2 bands), not a monthly time series
//   like your other datasets - deforestation history, not a repeating
//   monthly variable.
// - Band 0 (treecover2000): 0-100, % tree cover as of year 2000.
// - Band 1 (lossyear): 0 = no loss detected through the dataset's
//   coverage; 1-23 = year of loss (1=2001, 2=2002, ..., 18=2018).
//   For "disturbance through our 2003-2018 study period," you'll
//   filter for lossyear values 1-18 in the processing script.
// ============================================================