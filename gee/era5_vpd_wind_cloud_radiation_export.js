// ============================================================
// GEE Export Script: VPD components, wind, cloud, solar radiation
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. IMPORTANT: check the console for errors before exporting - I could
//    not verify these band names live. If any band errors, search the
//    GEE Data Catalog for "ERA5 monthly" and check the current band
//    list on that dataset's page - band names occasionally change
//    between ERA5 dataset versions.
// 5. Go to "Tasks" tab, run ALL FIVE export tasks
// 6. Files appear in Google Drive under "GEE_exports" once done
// ============================================================

var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);
Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

var startDate = '2003-01-01';
var endDate = '2018-12-31';

var era5 = ee.ImageCollection('ECMWF/ERA5/MONTHLY')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var months = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'month').round().subtract(1));

function monthlyExport(bandName, description, filePrefix, scaleFactor, offset) {
  scaleFactor = scaleFactor || 1;
  offset = offset || 0;
  var monthly = ee.ImageCollection(months.map(function(m) {
    var start = ee.Date(startDate).advance(m, 'month');
    var end = start.advance(1, 'month');
    var img = era5.filterDate(start, end).select(bandName).mean()
      .multiply(scaleFactor).add(offset).clip(region);
    return img.set('system:time_start', start.millis());
  }));
  var stack = monthly.toBands();
  Export.image.toDrive({
    image: stack,
    description: description,
    folder: 'GEE_exports',
    fileNamePrefix: filePrefix,
    region: region,
    scale: 27830,
    maxPixels: 1e10
  });
}

// ---- 1. Dewpoint temperature (for VPD calculation, paired with your
// existing mean_2m_air_temperature export) - Celsius ----
monthlyExport('dewpoint_2m_temperature', 'ERA5_dewpoint_export',
  'era5_dewpoint_amazon_cerrado_monthly', 1, -273.15);

// ---- 2. Wind components (u = east-west, v = north-south, m/s) ----
monthlyExport('u_component_of_wind_10m', 'ERA5_wind_u_export',
  'era5_wind_u_amazon_cerrado_monthly');
monthlyExport('v_component_of_wind_10m', 'ERA5_wind_v_export',
  'era5_wind_v_amazon_cerrado_monthly');

// ---- 3. Cloud cover (MODIS Terra Atmosphere Monthly, 1x1 degree - much
// coarser than your other datasets, so expect some smoothing at the
// patch level; verify 'Cloud_Fraction_Mean_Mean' is the exact band name
// in the console before exporting - band naming in this collection has
// historically included multiple similarly-named cloud fraction bands) ----
var modisCloud = ee.ImageCollection('MODIS/061/MOD08_M3')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var cloudMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var img = modisCloud.filterDate(start, end).select('Cloud_Fraction_Mean_Mean')
    .mean().clip(region);
  return img.set('system:time_start', start.millis());
}));
var cloudStack = cloudMonthly.toBands();
Export.image.toDrive({
  image: cloudStack,
  description: 'MODIS_cloud_fraction_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'modis_cloud_fraction_amazon_cerrado_monthly',
  region: region,
  scale: 111000, // MODIS atmosphere product native ~1 degree (~111km)
  maxPixels: 1e10
});

// ---- 4. Solar radiation - INCOMING (not net), ERA5-Land, ~11km native
// resolution, units J/m^2 (already a monthly accumulated sum) ----
var era5Land = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var solarMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var img = era5Land.filterDate(start, end).select('surface_solar_radiation_downwards_sum')
    .mean().clip(region);
  return img.set('system:time_start', start.millis());
}));
var solarStack = solarMonthly.toBands();
Export.image.toDrive({
  image: solarStack,
  description: 'ERA5Land_solar_radiation_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'era5land_solar_radiation_amazon_cerrado_monthly',
  region: region,
  scale: 11132,
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - VPD is not a direct band - it will be CALCULATED in Python from
//   temperature + dewpoint temperature (both now exported) using the
//   standard saturation-vapor-pressure formula. See the processing
//   script for the exact calculation.
// - Wind u/v are already in physical units (m/s), no scaling needed.
// - Cloud fraction (MODIS) is at ~1 degree resolution - much coarser
//   than everything else you've used (VODCA/CHIRPS/ERA5 are all much
//   finer). Each ~90km patch will only span a fraction of one MODIS
//   cell, so this variable will be much smoother/less spatially
//   detailed than your other drivers - worth noting in the write-up.
// - Solar radiation is INCOMING (downward) radiation in J/m^2 per
//   month, not net radiation - these measure different things, don't
//   mix them up.
// - All exports use the same region/date range as your other data, so
//   band 0 = Jan 2003 in every file, consistent with everything else.
// ============================================================