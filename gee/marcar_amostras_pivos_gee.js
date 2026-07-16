/****
 * Marcação interativa de pivôs e não pivôs no Google Earth Engine Code Editor.
 *
 * Uso:
 * 1. Execute o script.
 * 2. Desenhe um polígono com a ferramenta exibida no mapa.
 * 3. Clique em "Adicionar como PIVÔ" ou "Adicionar como NÃO PIVÔ".
 * 4. Repita em áreas variadas e exporte a coleção para um Asset.
 ****/

var CONFIG = {
  ano: 2025,
  municipio: 'Jussara',
  estado: 'Goias',
  municipios: 'FAO/GAUL/2015/level2'
};

var jussara = ee.FeatureCollection(CONFIG.municipios)
  .filter(ee.Filter.eq('ADM2_NAME', CONFIG.municipio))
  .filter(ee.Filter.eq('ADM1_NAME', CONFIG.estado));
var regiao = jussara.geometry();

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
var contadorPivos = 0;
var contadorNaoPivos = 0;
var status = ui.Label('Pivôs: 0 | Não pivôs: 0');

function atualizarStatus(mensagem) {
  status.setValue(
    'Pivôs: ' + contadorPivos + ' | Não pivôs: ' + contadorNaoPivos +
    (mensagem ? '\n' + mensagem : '')
  );
}

function limparRascunho() {
  rascunho.geometries().reset([]);
  drawingTools.setShape('polygon');
  drawingTools.draw();
}

function adicionarAmostra(classe) {
  if (rascunho.geometries().length() === 0) {
    atualizarStatus('Desenhe um polígono antes de adicionar.');
    return;
  }

  var geometria = rascunho.getEeObject();
  var rotulo = classe === 1 ? 'pivo' : 'nao_pivo';
  amostras.push(ee.Feature(geometria, {
    classe: classe,
    rotulo: rotulo,
    fonte: 'marcacao_manual',
    ano_referencia: CONFIG.ano
  }));

  if (classe === 1) {
    contadorPivos += 1;
    Map.addLayer(geometria, {color: '00ff00'}, 'Pivô ' + contadorPivos);
  } else {
    contadorNaoPivos += 1;
    Map.addLayer(geometria, {color: 'ff0000'}, 'Não pivô ' + contadorNaoPivos);
  }
  limparRascunho();
  atualizarStatus('Amostra adicionada. Continue desenhando.');
}

var assetId = ui.Textbox({
  placeholder: 'projects/SEU_PROJETO/assets/amostras_pivos_jussara',
  style: {stretch: 'horizontal'}
});

function validarAmostras() {
  if (contadorPivos === 0 || contadorNaoPivos === 0) {
    atualizarStatus('Marque pelo menos um pivô e um não pivô antes de exportar.');
    return false;
  }
  return true;
}

function exportarAsset() {
  if (!validarAmostras()) return;
  var destino = assetId.getValue().trim();
  if (destino.indexOf('projects/') !== 0 || destino.indexOf('/assets/') === -1) {
    atualizarStatus('Informe um Asset ID no formato projects/.../assets/...');
    return;
  }
  Export.table.toAsset({
    collection: ee.FeatureCollection(amostras),
    description: 'amostras_pivos_jussara',
    assetId: destino
  });
  atualizarStatus('Tarefa criada. Abra a aba Tasks e clique em RUN.');
}

function exportarDrive() {
  if (!validarAmostras()) return;
  Export.table.toDrive({
    collection: ee.FeatureCollection(amostras),
    description: 'amostras_pivos_jussara',
    folder: 'GEE_exports',
    fileFormat: 'GeoJSON'
  });
  atualizarStatus('Tarefa criada. Abra a aba Tasks e clique em RUN.');
}

var painel = ui.Panel({
  widgets: [
    ui.Label('Treinamento de identificação de pivôs', {fontWeight: 'bold', fontSize: '16px'}),
    ui.Label(
      'Desenhe polígonos pequenos e homogêneos. Inclua pivôs em diferentes fases ' +
      'da cultura e não pivôs como pastagem, mata, urbano, rios e lavouras comuns.'
    ),
    ui.Button('Adicionar como PIVÔ', function () { adicionarAmostra(1); }, false, {stretch: 'horizontal'}),
    ui.Button('Adicionar como NÃO PIVÔ', function () { adicionarAmostra(0); }, false, {stretch: 'horizontal'}),
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
print('Amostras serão exportadas com classe 1 (pivô) e classe 0 (não pivô).');
