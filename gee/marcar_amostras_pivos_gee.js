/****
 * Marcação interativa de NÃO PIVÔS no Google Earth Engine Code Editor.
 * Os pivôs positivos vêm do SHP importado na variável `final_pivos3`.
 *
 * Uso:
 * 1. Execute o script.
 * 2. Desenhe um polígono com a ferramenta exibida no mapa.
 * 3. Clique em "Adicionar como NÃO PIVÔ".
 * 4. Repita e exporte uma coleção combinada (SHP = 1; desenhos = 0).
 ****/

var CONFIG = {
  ano: 2025,
  municipio: 'Jussara',
  estado: 'Goias',
  municipios: 'FAO/GAUL/2015/level2',
  // Asset positivo informado para o projeto. Não requer importação manual.
  assetPivosPositivos: 'projects/sefazgogeoprocessamento/assets/final_pivos3'
};

var jussara = ee.FeatureCollection(CONFIG.municipios)
  .filter(ee.Filter.eq('ADM2_NAME', CONFIG.municipio))
  .filter(ee.Filter.eq('ADM1_NAME', CONFIG.estado));
var regiao = jussara.geometry();

// O Asset configurado é usado diretamente. Uma variável importada chamada
// `final_pivos3`, se existir, tem prioridade para permitir uma revisão alternativa.
var PIVOS_SHP = typeof final_pivos3 !== 'undefined'
  ? ee.FeatureCollection(final_pivos3)
  : ee.FeatureCollection(CONFIG.assetPivosPositivos);
PIVOS_SHP = PIVOS_SHP.filterBounds(regiao).map(function (feature) {
  return feature.set({
    classe: 1,
    rotulo: 'pivo',
    fonte: 'final_pivos3'
  });
});

function mascararS2(image) {
  var scl = image.select('SCL');
  var mascara = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));
  return image.updateMask(mascara)
    .select(['B2', 'B3', 'B4', 'B8'])
    .multiply(0.0001);
}

var inicio = ee.Date.fromYMD(CONFIG.ano, 1, 1);
var mosaico = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(regiao)
  .filterDate(inicio, inicio.advance(1, 'year'))
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(mascararS2)
  .median()
  .clip(regiao);
var ndvi = mosaico.normalizedDifference(['B8', 'B4']).rename('NDVI');

Map.centerObject(jussara, 11);
Map.addLayer(mosaico, {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.3}, 'RGB 2025');
Map.addLayer(ndvi, {min: 0, max: 0.9, palette: ['8b4513', 'ffff99', '006400']}, 'NDVI', false);
Map.addLayer(jussara.style({color: 'ffffff', fillColor: '00000000'}), {}, 'Limite de Jussara');
if (PIVOS_SHP) {
  Map.addLayer(PIVOS_SHP.style({color: '00ff00', fillColor: '00ff0033'}), {}, 'Pivôs do SHP');
}

var drawingTools = Map.drawingTools();
drawingTools.setShown(true);
drawingTools.setDrawModes(['polygon', 'rectangle']);
drawingTools.layers().reset();
var rascunho = ui.Map.GeometryLayer({
  geometries: null,
  // GeometryLayer aceita apenas nome alfanumérico iniciado por uma letra.
  name: 'DesenhoAtual',
  color: 'ffff00'
});
drawingTools.layers().add(rascunho);
drawingTools.setShape('polygon');
drawingTools.draw();

var amostras = [];
var contadorNaoPivos = 0;
var status = ui.Label('Não pivôs marcados: 0');

function atualizarStatus(mensagem) {
  status.setValue(
    'SHP final_pivos3: ' + (PIVOS_SHP ? 'carregado' : 'não importado') +
    ' | Não pivôs marcados: ' + contadorNaoPivos +
    (mensagem ? '\n' + mensagem : '')
  );
}

function limparRascunho() {
  rascunho.geometries().reset([]);
  drawingTools.setShape('polygon');
  drawingTools.draw();
}

function adicionarNaoPivo() {
  if (rascunho.geometries().length() === 0) {
    atualizarStatus('Desenhe um polígono antes de adicionar.');
    return;
  }

  var geometria = rascunho.getEeObject();
  amostras.push(ee.Feature(geometria, {
    classe: 0,
    rotulo: 'nao_pivo',
    fonte: 'marcacao_manual',
    ano_referencia: CONFIG.ano
  }));

  contadorNaoPivos += 1;
  Map.addLayer(geometria, {color: 'ff0000'}, 'Não pivô ' + contadorNaoPivos);
  limparRascunho();
  atualizarStatus('Amostra adicionada. Continue desenhando.');
}

var assetId = ui.Textbox({
  placeholder: 'projects/SEU_PROJETO/assets/final_pivos3_rotulado',
  style: {stretch: 'horizontal'}
});

function validarAmostras() {
  if (!PIVOS_SHP) {
    atualizarStatus('Importe o SHP no script com o nome final_pivos3 antes de exportar.');
    return false;
  }
  if (contadorNaoPivos === 0) {
    atualizarStatus('Marque pelo menos um não pivô antes de exportar.');
    return false;
  }
  return true;
}

function colecaoRotulada() {
  return PIVOS_SHP.merge(ee.FeatureCollection(amostras));
}

function exportarAsset() {
  if (!validarAmostras()) return;
  // getValue() retorna undefined enquanto o campo ainda não foi preenchido.
  var destino = String(assetId.getValue() || '').trim();
  if (destino.indexOf('projects/') !== 0 || destino.indexOf('/assets/') === -1) {
    atualizarStatus('Informe um Asset ID no formato projects/.../assets/...');
    return;
  }
  Export.table.toAsset({
    collection: colecaoRotulada(),
    description: 'amostras_pivos_jussara',
    assetId: destino
  });
  atualizarStatus('Tarefa criada. Abra a aba Tasks e clique em RUN.');
}

function exportarDrive() {
  if (!validarAmostras()) return;
  Export.table.toDrive({
    collection: colecaoRotulada(),
    description: 'amostras_pivos_jussara',
    folder: 'GEE_exports',
    fileFormat: 'GeoJSON'
  });
  atualizarStatus('Tarefa criada. Abra a aba Tasks e clique em RUN.');
}

var painel = ui.Panel({
  widgets: [
    ui.Label('Marcar objetos que NÃO são pivôs', {fontWeight: 'bold', fontSize: '16px'}),
    ui.Label(
      'Os pivôs positivos são carregados do SHP final_pivos3. Desenhe polígonos ' +
      'pequenos em pastagem, mata, urbano, rios e lavouras comuns.'
    ),
    ui.Button('Adicionar como NÃO PIVÔ', adicionarNaoPivo, false, {stretch: 'horizontal'}),
    ui.Button('Limpar desenho atual', limparRascunho, false, {stretch: 'horizontal'}),
    status,
    ui.Label('Asset ID de saída:'),
    assetId,
    ui.Button('Criar tarefa de exportação para Asset', exportarAsset, false, {stretch: 'horizontal'}),
    ui.Button('Criar tarefa de exportação para Drive', exportarDrive, false, {stretch: 'horizontal'})
  ],
  style: {width: '360px', padding: '8px'}
});

ui.root.insert(0, painel);
atualizarStatus(
  'SHP de pivôs carregado. Desenhe agora somente exemplos de não pivô.'
);
print('Amostras serão exportadas com classe 1 (pivô) e classe 0 (não pivô).');
