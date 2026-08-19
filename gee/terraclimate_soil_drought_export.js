// ============================================================
// GEE Export Script: TerraClimate Soil Moisture + PDSI Drought Index
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. IMPORTANT: check the console for errors loading the
//    IDAHO_EPSCOR/TERRACLIMATE collection and the 'soil'/'pdsi' bands -
//    I could not verify this live, so confirm the collection ID and
//    band names still match the current GEE Data Catalog
//    (developers.google.com/earth-engine/datasets) before exporting.
//    ALSO VERIFY the scale factors below (soil x0.1, pdsi x0.01) against
//    the catalog page's band description - these are from memory and
//    may need correcting.
// 5. Go to "Tasks" tab, run BOTH export tasks
// 6. Files appear in Google Drive under "GEE_exports" once done
// ============================================================

// ---- 1. Study region - SAME as VODCA/CHIRPS/ERA5 exports ----
var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

// ---- 2. Time range - SAME as all other exports ----
var startDate = '2003-01-01';
var endDate = '2018-12-31';

var terraclimate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var months = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'month').round().subtract(1));

// ============================================================
// PART A: Soil moisture ('soil' band, scale factor 0.1 -> mm)
// ============================================================
var soilMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var img = terraclimate.filterDate(start, end)
    .select('soil')
    .mean()
    .multiply(0.1)  // VERIFY this scale factor in the catalog
    .clip(region);
  return img.set('system:time_start', start.millis());
}));
var soilStack = soilMonthly.toBands();

Export.image.toDrive({
  image: soilStack,
  description: 'TerraClimate_soil_moisture_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'terraclimate_soil_moisture_amazon_cerrado_monthly',
  region: region,
  scale: 4638, // TerraClimate native resolution ~4km (2.5 arcmin)
  maxPixels: 1e10
});

// ============================================================
// PART B: PDSI drought index ('pdsi' band, scale factor 0.01)
// ============================================================
var pdsiMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var img = terraclimate.filterDate(start, end)
    .select('pdsi')
    .mean()
    .multiply(0.01)  // VERIFY this scale factor in the catalog
    .clip(region);
  return img.set('system:time_start', start.millis());
}));
var pdsiStack = pdsiMonthly.toBands();

Export.image.toDrive({
  image: pdsiStack,
  description: 'TerraClimate_pdsi_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'terraclimate_pdsi_amazon_cerrado_monthly',
  region: region,
  scale: 4638,
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - PDSI (Palmer Drought Severity Index): negative = drought,
//   positive = wet conditions, roughly -10 to +10 range typically.
// - Soil moisture ('soil') is in mm of water in the soil column.
// - Both use the same band ordering (band 0 = Jan 2003, ..., band 191
//   = Dec 2018) as your other exports, so they line up by month index.
// ============================================================