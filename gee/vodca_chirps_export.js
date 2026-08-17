// ============================================================
// GEE Export Script: VODCA (vegetation) + CHIRPS (precipitation)
// Study area: Amazon-Cerrado Transition (northern Mato Grosso / 
// southern Pará), working boundary box
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. Go to the "Tasks" tab (top right) - you'll see 2 export
//    tasks waiting ("VODCA_monthly_export" and "CHIRPS_monthly_export")
// 5. Click "Run" next to each task to start it
// 6. Files will appear in your Google Drive under a folder
//    called "GEE_exports" once done (can take a while - large area/time range)
// ============================================================

// ---- 1. Define study region (adjust later if needed) ----
var region = ee.Geometry.Rectangle([-58, -14, -50, -8]); 
// [west, south, east, north] = [-58W, -14S, -50W, -8S]

Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

// ---- 2. Define time range ----
// VODCA C-band covers 2002-2018, CHIRPS covers 1981-present.
// Using the overlapping range where both exist, extended to CHIRPS max:
var startDate = '2003-01-01';
var endDate = '2018-12-31';

// ============================================================
// PART A: VODCA (vegetation optical depth) - monthly aggregation
// ============================================================

// C-band VODCA v2 collection (from GEE community catalog)
var vodca = ee.ImageCollection('projects/sat-io/open-datasets/VODCA/CKXU_BAND_V2')
  .filterDate(startDate, endDate)
  .filterBounds(region);

// Build a list of year-month pairs to aggregate over
var months = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'month').round().subtract(1));

var vodcaMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var monthMean = vodca.filterDate(start, end).mean().clip(region);
  return monthMean.set('system:time_start', start.millis())
                   .set('month_label', start.format('YYYY_MM'));
}));

// Convert the whole monthly time series into a single multi-band image
// (each band = one month), so we only need ONE export task
var vodcaStack = vodcaMonthly.toBands();

Export.image.toDrive({
  image: vodcaStack,
  description: 'VODCA_monthly_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'vodca_amazon_cerrado_monthly',
  region: region,
  scale: 25000, // ~0.25 degree native resolution
  maxPixels: 1e10
});

// ============================================================
// PART B: CHIRPS (precipitation) - monthly aggregation
// ============================================================

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var chirpsMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  // Precipitation should be SUMMED over the month, not averaged
  var monthSum = chirps.filterDate(start, end).sum().clip(region);
  return monthSum.set('system:time_start', start.millis())
                  .set('month_label', start.format('YYYY_MM'));
}));

var chirpsStack = chirpsMonthly.toBands();

Export.image.toDrive({
  image: chirpsStack,
  description: 'CHIRPS_monthly_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'chirps_amazon_cerrado_monthly',
  region: region,
  scale: 5000, // ~0.05 degree native resolution
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - Each band in the exported GeoTIFF corresponds to one month.
//   Band names will look like "0_VOD" (month 0), "1_VOD" (month 1), etc.
//   Month 0 = startDate's month, counting up from there.
// - VODCA resolution (~25km) is coarser than CHIRPS (~5km) - this is
//   normal and expected; you'll aggregate CHIRPS up to match your
//   final patch size later, not the reverse.
// - If the export fails or times out due to file size, reduce the
//   date range (e.g. do 5 years at a time) and run multiple exports.
// ============================================================