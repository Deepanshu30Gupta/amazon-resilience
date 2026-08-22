// ============================================================
// GEE Export Script: MODIS Land Surface Temperature (for Canopy vs
// Ambient Temperature / deltaT calculation)
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. Check console for errors before exporting - I could not verify
//    this live. If 'MODIS/061/MOD11A2' errors, search the GEE Data
//    Catalog for "MODIS land surface temperature" for the current ID.
// 5. Go to "Tasks" tab, run the export task
// 6. File appears in Google Drive under "GEE_exports" once done
// ============================================================

var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);
Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

var startDate = '2003-01-01';
var endDate = '2018-12-31';

// MOD11A2: 8-day composite LST, 1km resolution. We average the 8-day
// composites within each month to get a monthly value.
var modisLST = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var months = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'month').round().subtract(1));

// LST_Day_1km band: scale factor 0.02, units Kelvin - convert to Celsius
var lstMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var img = modisLST.filterDate(start, end).select('LST_Day_1km').mean()
    .multiply(0.02).subtract(273.15).clip(region);
  return img.set('system:time_start', start.millis());
}));

var lstStack = lstMonthly.toBands();

Export.image.toDrive({
  image: lstStack,
  description: 'MODIS_LST_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'modis_lst_amazon_cerrado_monthly',
  region: region,
  scale: 1000,
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - LST_Day_1km is the daytime land surface (canopy) temperature,
//   already converted to Celsius here.
// - "Canopy vs Ambient Temperature" (deltaT) will be CALCULATED in
//   Python as: deltaT = LST (this file) - 2m air temperature (your
//   existing era5_temperature file). Positive deltaT means the canopy
//   surface is warmer than the surrounding air - a known indicator of
//   plant water stress (stomatal closure reduces evaporative cooling).
// - MOD11A2 has real data gaps from persistent cloud cover (LST can't
//   be retrieved through clouds) - expect more missing data here than
//   in your other datasets, especially in the wettest months/areas.
// ============================================================