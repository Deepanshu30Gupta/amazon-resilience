// ============================================================
// GEE Export Script: Root-Zone Soil Moisture (RZSM)
// Study area: Amazon-Cerrado Transition, v2 expanded region
// ============================================================
// HOW TO USE:
// 1. Go to https://code.earthengine.google.com
// 2. Paste this whole script into the code editor
// 3. Click "Run"
// 4. Check console for errors before exporting (should be low-risk -
//    this reuses the ECMWF/ERA5_LAND/MONTHLY_AGGR collection that
//    already worked for your solar radiation export)
// 5. Go to "Tasks" tab, run the export task
// 6. File appears in Google Drive under "GEE_exports" once done
// ============================================================

var region = ee.Geometry.Rectangle([-66, -18, -46, -4]);
Map.centerObject(region, 6);
Map.addLayer(region, {color: 'red'}, 'Study Area');

var startDate = '2003-01-01';
var endDate = '2018-12-31';

var era5Land = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate(startDate, endDate)
  .filterBounds(region);

var months = ee.List.sequence(0, ee.Date(endDate).difference(ee.Date(startDate), 'month').round().subtract(1));

// ERA5-Land has 4 soil depth layers:
//   layer_1: 0-7cm (surface - NOT root zone, already have TerraClimate's
//            surface soil moisture as a separate variable)
//   layer_2: 7-28cm   \  these two together are the standard
//   layer_3: 28-100cm /  "root zone" depth used in most studies
//   layer_4: 100-289cm (deep zone, excluded here - could add later if wanted)
// RZSM here = average of layer_2 and layer_3, in m^3/m^3 (volumetric
// water content, already physical units, no scaling needed)

var rzsmMonthly = ee.ImageCollection(months.map(function(m) {
  var start = ee.Date(startDate).advance(m, 'month');
  var end = start.advance(1, 'month');
  var layer2 = era5Land.filterDate(start, end).select('volumetric_soil_water_layer_2').mean();
  var layer3 = era5Land.filterDate(start, end).select('volumetric_soil_water_layer_3').mean();
  var rzsm = layer2.add(layer3).divide(2).rename('rzsm').clip(region);
  return rzsm.set('system:time_start', start.millis());
}));

var rzsmStack = rzsmMonthly.toBands();

Export.image.toDrive({
  image: rzsmStack,
  description: 'ERA5Land_RZSM_export',
  folder: 'GEE_exports',
  fileNamePrefix: 'era5land_rzsm_amazon_cerrado_monthly',
  region: region,
  scale: 11132,
  maxPixels: 1e10
});

// ============================================================
// NOTES:
// - Values are volumetric soil water content (m^3/m^3), typically
//   0-0.5 range for most soils - no scale factor conversion needed.
// - This is DISTINCT from TerraClimate's 'soil' variable already used
//   (which is a modeled total soil water storage in mm, closer to
//   surface-weighted) - RZSM specifically targets the 7-100cm depth
//   most relevant to tree root water uptake.
// - Same region/date range as everything else, band 0 = Jan 2003.
// ============================================================