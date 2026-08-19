// ============================================================
// GEE Export Script: ERA5 Monthly Temperature
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. IMPORTANT: before running the export, check the console/inspector
//    for any errors loading the ECMWF/ERA5/MONTHLY collection - GEE's
//    catalog changes over time, so confirm this collection ID and the
//    'mean_2m_air_temperature' band name still exist. If it errors,
//    search the GEE Data Catalog (developers.google.com/earth-engine/datasets)
//    for "ERA5 monthly" and swap in the current collection ID/band name.
// 5. Go to the "Tasks" tab (top right), click "Run" on the export task
// 6. File will appear in your Google Drive under "GEE_exports" once done
// ============================================================

// ---- 1. Study region - SAME as the v2 VODCA/CHIRPS export ----
var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);
// [west, south, east, north] = [-66W, -18S, -46W, -4S]

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

// ---- 2. Time range - SAME as VODCA/CHIRPS, so months line up exactly ----
var startDate = '2003-01-01';
var endDate = '2018-12-31';

// ============================================================
// ERA5 monthly temperature
// ============================================================

var era5 = ee.ImageCollection('ECMWF/ERA5/MONTHLY')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var months = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'month').round().subtract(1));

var tempMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var monthImg = era5.filterDate(start, end)
    .select('mean_2m_air_temperature')
    .mean()
    .subtract(273.15)  // Kelvin -> Celsius
    .clip(region);
  return monthImg.set('system:time_start', start.millis())
                  .set('month_label', start.format('YYYY_MM'));
}));

var tempStack = tempMonthly.toBands();

Export.image.toDrive({
  image: tempStack,
  description: 'ERA5_temperature_monthly_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'era5_temperature_amazon_cerrado_monthly',
  region: region,
  scale: 27830, // ERA5 native resolution is ~0.25 degree (~27.8km)
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - Each band = one month, same ordering as the VODCA/CHIRPS exports
//   (band 0 = Jan 2003, ..., band 191 = Dec 2018), so this file lines
//   up directly with your existing patch_timeseries.csv by month index.
// - Values are in Celsius (converted from the source Kelvin).
// - ERA5 resolution (~27.8km) is close to VODCA's (~25km) - similar
//   aggregation behavior to what you already have for VOD.
// - If 'ECMWF/ERA5/MONTHLY' has been deprecated/replaced by the time
//   you run this, try 'ECMWF/ERA5_LAND/MONTHLY_AGGR' with band
//   'temperature_2m' instead (also in Kelvin, same conversion needed).
// ============================================================