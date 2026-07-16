/****
 * Detecção de pivôs centrais por textura e forma em Jussara-GO (Sentinel-2, 2025)
 *
 * Cole este script no Google Earth Engine Code Editor. A área municipal usa uma
 * coleção pública; somente as amostras positivas precisam ser informadas.
 ****/

// -----------------------------------------------------------------------------
// 1. CONFIGURAÇÃO
// -----------------------------------------------------------------------------
var CONFIG = {
  ano: 2025,
  municipioNome: 'Jussara',

  // Limites administrativos públicos disponíveis no catálogo do Earth Engine.
  municipios: 'FAO/GAUL/2015/level2',
  campoMunicipio: 'ADM2_NAME',
  campoEstado: 'ADM1_NAME',
  estadoNome: 'Goias',

  // Informe um asset real OU use as amostras inline definidas logo abaixo.
  // Nunca deixe aqui o texto de exemplo "users/SEU_USUARIO/...".
  // Se você usou marcar_amostras_pivos_gee.js, informe o asset combinado aqui;
  // ele deve conter a propriedade `classe` (1 = pivô; 0 = não pivô).
  assetAmostrasRotuladas: null,
  assetPivosPositivos: null,
  usarAmostrasPositivasInline: false,

  // Sem amostras positivas, o script muda automaticamente para o modo não
  // supervisionado e procura bordas/variação dentro de uma janela circular.
  raioCircularPixels: 35,
  limiarCanny: 0.08,
  densidadeBordaMinima: 0.025,
  densidadeBordaMaxima: 0.35,
  desvioNdviMinimo: 0.035,

  // Opcional: substitua por polígonos/pontos de não pivô (pastagem, mata, urbano etc.).
  // Com null, o script gera amostras negativas aleatórias. Um caminho inexistente
  // sempre produz Collection.loadTable; portanto, não use um placeholder aqui.
  assetAmostrasNegativas: null,

  escala: 10,
  quantidadeNegativos: 700,
  probabilidadeTreino: 0.7,
  sementes: {
    negativos: 42,
    treinoTeste: 13,
    randomForest: 27
  },

  bandasPreditoras: [
    'B2', 'B3', 'B4', 'B8', 'NDVI', 'NDWI',
    'B8_contrast', 'B8_corr', 'B8_ent', 'NDVI_contrast', 'NDVI_corr', 'NDVI_ent',
    'B8_stdDev_3', 'B8_stdDev_7', 'NDVI_stdDev_3', 'NDVI_stdDev_7'
  ]
};

// Alternativa ao asset: cole aqui pontos ou polígonos desenhados/importados no
// Code Editor e altere `usarAmostrasPositivasInline` para true. Exemplo:
// ee.Feature(ee.Geometry.Point([-50.0, -15.0]))
var AMOSTRAS_POSITIVAS_INLINE = ee.FeatureCollection([
  // ee.Feature(geometry1),
  // ee.Feature(geometry2)
]);

var modoSupervisionado = Boolean(
  CONFIG.assetAmostrasRotuladas ||
  CONFIG.assetPivosPositivos ||
  CONFIG.usarAmostrasPositivasInline
);

// -----------------------------------------------------------------------------
// 2. ÁREA DE ESTUDO E AMOSTRAS
// -----------------------------------------------------------------------------
var municipios = ee.FeatureCollection(CONFIG.municipios);
var jussara = municipios
  .filter(ee.Filter.eq(CONFIG.campoMunicipio, CONFIG.municipioNome))
  .filter(ee.Filter.eq(CONFIG.campoEstado, CONFIG.estadoNome));
var geometriaJussara = jussara.geometry();

var amostrasRotuladas = CONFIG.assetAmostrasRotuladas
  ? ee.FeatureCollection(CONFIG.assetAmostrasRotuladas).filterBounds(geometriaJussara)
  : null;

var colecaoPivos = modoSupervisionado
  ? (amostrasRotuladas
    ? amostrasRotuladas.filter(ee.Filter.eq('classe', 1))
    : (CONFIG.assetPivosPositivos
    ? ee.FeatureCollection(CONFIG.assetPivosPositivos)
    : AMOSTRAS_POSITIVAS_INLINE))
  : ee.FeatureCollection([]);

var pivos = colecaoPivos
  .filterBounds(geometriaJussara)
  .map(function (feature) {
    return feature.set('classe', 1);
  });

var negativos = ee.FeatureCollection([]);
if (modoSupervisionado) {
  negativos = amostrasRotuladas
    ? amostrasRotuladas.filter(ee.Filter.eq('classe', 0))
    : (CONFIG.assetAmostrasNegativas
    ? ee.FeatureCollection(CONFIG.assetAmostrasNegativas)
      .filterBounds(geometriaJussara)
      .map(function (feature) { return feature.set('classe', 0); })
    : ee.FeatureCollection.randomPoints({
      region: geometriaJussara.difference(pivos.geometry().buffer(250), 1),
      points: CONFIG.quantidadeNegativos,
      seed: CONFIG.sementes.negativos,
      maxError: 10
    }).map(function (feature) { return feature.set('classe', 0); }));
}

var amostras = pivos.merge(negativos);

// -----------------------------------------------------------------------------
// 3. MOSAICO SENTINEL-2 LIVRE DE NUVENS
// -----------------------------------------------------------------------------
function mascararS2Sr(image) {
  var scl = image.select('SCL');
  var mascaraScl = scl.neq(3)   // sombra de nuvem
    .and(scl.neq(8))            // nuvem média probabilidade
    .and(scl.neq(9))            // nuvem alta probabilidade
    .and(scl.neq(10))           // cirrus
    .and(scl.neq(11));          // neve/gelo

  var qa = image.select('QA60');
  var mascaraQa = qa.bitwiseAnd(1 << 10).eq(0)
    .and(qa.bitwiseAnd(1 << 11).eq(0));

  return image.updateMask(mascaraScl.and(mascaraQa))
    .select(['B2', 'B3', 'B4', 'B8'])
    .multiply(0.0001)
    .copyProperties(image, ['system:time_start']);
}

var inicio = ee.Date.fromYMD(CONFIG.ano, 1, 1);
var fim = inicio.advance(1, 'year');

var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(geometriaJussara)
  .filterDate(inicio, fim)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(mascararS2Sr);

var mosaico = sentinel2.median().clip(geometriaJussara);
var ndvi = mosaico.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = mosaico.normalizedDifference(['B3', 'B8']).rename('NDWI');
var base = mosaico.addBands([ndvi, ndwi]);

// -----------------------------------------------------------------------------
// 4. MÉTRICAS DE TEXTURA E FORMATO
// -----------------------------------------------------------------------------
function texturaGlcm(image, banda) {
  return image.select(banda)
    .unitScale(-0.2, 0.9)
    .multiply(255)
    .toUint8()
    .glcmTexture({size: 5})
    .select([
      banda + '_contrast',
      banda + '_corr',
      banda + '_ent'
    ]);
}

function desvioPadrao(image, banda, raioPixels) {
  return image.select(banda)
    .reduceNeighborhood({
      reducer: ee.Reducer.stdDev(),
      kernel: ee.Kernel.circle({radius: raioPixels, units: 'pixels'}),
      skipMasked: true
    })
    .rename(banda + '_stdDev_' + raioPixels);
}

var textura = texturaGlcm(base, 'B8')
  .addBands(texturaGlcm(base, 'NDVI'))
  .addBands(desvioPadrao(base, 'B8', 3))
  .addBands(desvioPadrao(base, 'B8', 7))
  .addBands(desvioPadrao(base, 'NDVI', 3))
  .addBands(desvioPadrao(base, 'NDVI', 7));

var preditores = base.addBands(textura).select(CONFIG.bandasPreditoras);

// -----------------------------------------------------------------------------
// 5. CLASSIFICAÇÃO SUPERVISIONADA OU DETECÇÃO CIRCULAR NÃO SUPERVISIONADA
// -----------------------------------------------------------------------------
var classificado;

if (modoSupervisionado) {
  print('Modo de detecção', 'Supervisionado (Random Forest)');
  var dadosTreinamento = preditores.sampleRegions({
    collection: amostras,
    properties: ['classe'],
    scale: CONFIG.escala,
    tileScale: 4,
    geometries: true
  }).randomColumn('aleatorio', CONFIG.sementes.treinoTeste);

  var treino = dadosTreinamento.filter(ee.Filter.lt('aleatorio', CONFIG.probabilidadeTreino));
  var teste = dadosTreinamento.filter(ee.Filter.gte('aleatorio', CONFIG.probabilidadeTreino));

  var classificador = ee.Classifier.smileRandomForest({
    numberOfTrees: 250,
    variablesPerSplit: 5,
    minLeafPopulation: 2,
    bagFraction: 0.65,
    seed: CONFIG.sementes.randomForest
  }).train({
    features: treino,
    classProperty: 'classe',
    inputProperties: CONFIG.bandasPreditoras
  });

  var matrizConfusao = teste.classify(classificador)
    .errorMatrix('classe', 'classification');
  print('Matriz de confusão', matrizConfusao);
  print('Acurácia global', matrizConfusao.accuracy());
  print('Kappa', matrizConfusao.kappa());
  print('Importância das variáveis', classificador.explain().get('importance'));
  classificado = preditores.classify(classificador).rename('pivo');
} else {
  print(
    'Modo de detecção',
    'Não supervisionado: correlação espacial em janela circular (sem matriz de confusão)'
  );

  var bordasNdvi = ee.Algorithms.CannyEdgeDetector({
    image: ndvi,
    threshold: CONFIG.limiarCanny,
    sigma: 1
  });
  var kernelCircular = ee.Kernel.circle({
    radius: CONFIG.raioCircularPixels,
    units: 'pixels',
    normalize: true
  });
  var densidadeBordas = bordasNdvi.reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: kernelCircular,
    skipMasked: true
  }).rename('densidade_bordas_circular');
  var variacaoCircular = ndvi.reduceNeighborhood({
    reducer: ee.Reducer.stdDev(),
    kernel: kernelCircular,
    skipMasked: true
  }).rename('variacao_ndvi_circular');

  classificado = densidadeBordas.gte(CONFIG.densidadeBordaMinima)
    .and(densidadeBordas.lte(CONFIG.densidadeBordaMaxima))
    .and(variacaoCircular.gte(CONFIG.desvioNdviMinimo))
    .rename('pivo');

  Map.addLayer(
    densidadeBordas,
    {min: 0, max: CONFIG.densidadeBordaMaxima, palette: ['000000', 'ffff00', 'ff0000']},
    'Diagnóstico: densidade de bordas circular',
    false
  );
}
var pivosLimpos = classificado.eq(1)
  .focal_min({radius: 1, units: 'pixels'})
  .focal_max({radius: 2, units: 'pixels'})
  .selfMask()
  .rename('pivos_centrais');

// -----------------------------------------------------------------------------
// 6. VISUALIZAÇÃO E EXPORTAÇÃO
// -----------------------------------------------------------------------------
Map.centerObject(jussara, 11);
Map.addLayer(base, {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.3}, 'Sentinel-2 RGB 2025');
Map.addLayer(pivos, {color: '00ff00'}, 'Amostras positivas');
Map.addLayer(negativos, {color: 'ff0000'}, 'Amostras negativas');
Map.addLayer(pivosLimpos, {palette: ['00ffff']}, 'Pivôs classificados por textura/forma');

Export.image.toDrive({
  image: pivosLimpos.toByte(),
  description: 'pivos_textura_forma_jussara_' + CONFIG.ano,
  folder: 'GEE_exports',
  fileNamePrefix: 'pivos_textura_forma_jussara_' + CONFIG.ano,
  region: geometriaJussara,
  scale: CONFIG.escala,
  maxPixels: 1e13
});
