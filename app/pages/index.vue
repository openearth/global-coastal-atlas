<script setup lang="ts">
import { MapboxMap, MapboxLayer } from '@studiometa/vue-mapbox-gl'
import MapboxDraw from '@mapbox/mapbox-gl-draw'
import 'mapbox-gl/dist/mapbox-gl.css'
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css'
import groupBy from 'lodash/groupBy'
import mean from 'lodash/mean'
import notebookTemplate from '~/assets/sliced_dataset_workbench.ipynb?raw'

import itemShape from '../../STAC/data/current/sub_threat/eapa-mapbox/eapa-mapbox-time-2010.json'
import catalogShape from '../../STAC/data/current/catalog.json'
import collectionShape from '../../STAC/data/current/sub_threat/collection.json'
import { getDataByPolygon } from '~/utils/zarr'

import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart } from 'echarts/charts'
import { EChartsOption } from 'echarts/types/dist/echarts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
} from 'echarts/components'
import VChart, { THEME_KEY } from 'vue-echarts'

type ItemType = typeof itemShape
type CatalogType = typeof catalogShape
type CollectionType = typeof collectionShape

type Option =
  | {
      label: string
      value: number
    }
  | {
      label: string
      value: string
    }

let url = useRequestURL()

let {
  public: { mapboxToken, pdfEndpoint },
} = useRuntimeConfig()

let baseURL = url.protocol + '//' + url.host + '/stac'

let catalogPath = `${baseURL}/catalog.json`

let headers = useRequestHeaders()

let { data: catalogJson } = await useFetch<CatalogType>(catalogPath, {
  headers,
})

let catalog = catalogJson.value

let childrenLinks = catalog?.links.filter((link) => link.rel === 'child') ?? []

let collections = (
  await Promise.all(
    childrenLinks.map(async (childLink) => {
      let { data } = await useFetch<CollectionType>(
        `${baseURL}/${childLink.href}`,
        {
          headers,
        },
      )
      return data.value
    }),
  )
).filter((collection) => collection?.links.some((link) => link.rel === 'item'))

let activeCollectionId = ref(collections[0]?.id)

let activeCollection = computed(() => {
  return collections.find(
    (collection) => collection?.id === activeCollectionId.value,
  )
})

let summaries = computed(() => {
  return activeCollection.value?.summaries
})

// let { data: activeCollection } = await useFetch(currentCollectionPath);

// let activeCollection = ref(currentCollection);

let variables = ref(
  Object.entries(summaries.value ?? {}).reduce((acc, [key, summary]) => {
    return {
      ...acc,
      [key]: summary.options[0],
    }
  }, {} as Record<string, Option>),
)

watchEffect(
  () => {
    variables.value = Object.entries(summaries.value ?? {}).reduce(
      (acc, [key, summary]) => {
        return {
          ...acc,
          [key]: summary.options[0],
        }
      },
      {} as Record<string, Option>,
    )
  },
  { flush: 'pre' },
)

let activeItemUrl = computed(() => {
  if (!activeCollection.value) return
  let foundLink =
    activeCollection.value.links
      .filter((l) => l.rel === 'item')
      .find((link) => {
        return Object.entries(variables.value).every(
          ([key, option]) =>
            link.properties?.[key as keyof typeof link.properties] ===
            option.value,
        )
      }) ?? activeCollection.value.links.filter((l) => l.rel === 'item')[0]

  return foundLink?.href
})

let { data } = await useAsyncData(
  () =>
    $fetch(`${baseURL}/${activeCollectionId.value}/${activeItemUrl.value}`, {
      headers,
    }),
  { watch: [activeItemUrl] },
)

let geojson = computed(() => {
  // let item = JSON.parse(data.value);
  let item = data.value

  if (!item?.assets) return {}

  let { mapbox } = item.assets
  let { properties } = item

  // if (visual) {
  //   return {
  //     id,
  //     type: "raster",
  //     source: {
  //       type: "raster",
  //       tiles: [visual.href],
  //       tileSize,
  //     },
  //   };
  // }

  return {
    id: item.id,
    type: properties['deltares:type'],
    source: {
      type: mapbox.type,
      url: mapbox.href,
    },
    'source-layer': mapbox.source,
    paint: properties['deltares:paint'],
  }
})

const option = ref(null)
let draw = ref<MapboxDraw>(null)
let polygons = ref([])

function instantiateDraw(map) {
  draw = new MapboxDraw({
    displayControlsDefault: false,
    controls: {
      polygon: true,
      trash: true,
    },
    defaultMode: 'draw_polygon',
  })

  map.addControl(draw)

  map.on('draw.create', updateArea)
  map.on('draw.delete', updateArea)
  map.on('draw.update', updateArea)

  async function updateArea(e) {
    const data = draw.getAll()
    polygons.value = data.features

    let polygonJson = encodeURIComponent(
      JSON.stringify(draw.getAll().features[0].geometry),
    )

    pdfLink.value = `${pdfEndpoint}?polygon=${polygonJson}`

    return

    let { allData, indices, rpValues, gwlValues } = await getDataByPolygon(data)

    let dimensions = ['rp', 'gwl', 'ensemble', 'nstations']

    function flattenArray(
      arr: any[],
      props: Record<string, any>,
      dimension: number,
    ) {
      return arr.reduce((acc, curr, index) => {
        return acc.concat(
          Array.isArray(curr) || curr instanceof Float64Array
            ? flattenArray(
                [...curr],
                {
                  ...props,
                  [dimensions[dimension]]: index,
                },
                dimension + 1,
              )
            : indices.includes(index)
            ? {
                ...props,
                [dimensions[dimension]]: index,
                value: curr,
              }
            : {},
        )
      }, [])
    }

    let rows = flattenArray(allData.data, {}, 0)

    let bucketDimension = 'gwl'
    let aggregateBy = 'rp'
    let aggregateFunction = mean

    let groupedBuckets = Object.values(groupBy(rows, bucketDimension))
    let groupedByAggregate = groupedBuckets.map((values) =>
      Object.values(groupBy(values, aggregateBy)),
    )
    let aggregated = groupedByAggregate.map((values) =>
      values.map((nestedValues) =>
        aggregateFunction(nestedValues.map((item) => item.value)),
      ),
    )

    let series: EChartsOption['series'] = aggregated.map((values, index) => {
      return {
        name: gwlValues[index],
        type: 'line',
        data: values,
      }
    })
    // Transform data into ECharts format
    // for (const [rp, gwlArray] of Object.entries(allData.data)) {
    //   for (const [gwl, allValues] of Object.entries(gwlArray)) {
    //     if (!series.some((item) => item.name === gwlValues[gwl])) {
    //       series.push({
    //         name: gwlValues[gwl],
    //         type: 'line',
    //         data: [],
    //       })
    //     }

    //     let values = indices.map((i) => allValues[i])
    //     let gwlIndex = series.findIndex((item) => item.name === gwlValues[gwl])
    //     series[gwlIndex].data.push(
    //       values.reduce((acc, curr) => acc + curr, 0) / values.length,
    //     )
    //   }
    // }

    option.value = {
      title: {
        text: 'ESL',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        // formatter: "{a} <br/>{b} : {c} ({d}%)"
      },
      xAxis: {
        type: 'category',
        data: [...rpValues],
      },
      yAxis: {
        type: 'value',
      },
      legend: {
        orient: 'horizontal',
        left: 'center',
        data: [...gwlValues],
      },
      series,
    }

    // if (data.features.length > 0) {
    //   const area = turf.area(data)
    //   // Restrict the area to 2 decimal points.
    //   const rounded_area = Math.round(area * 100) / 100
    //   answer.innerHTML = `<p><strong>${rounded_area}</strong></p><p>square meters</p>`
    // } else {
    //   answer.innerHTML = ''
    //   if (e.type !== 'draw.delete') alert('Click the map to draw a polygon.')
    // }
  }
}

let chartInit = {
  height: '400px',
}

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  PieChart,
])

provide(THEME_KEY, 'light')

async function downloadNotebook() {
  let polygonJson = JSON.stringify(
    draw.getAll().features[0].geometry,
  ).replaceAll('"', '\\"')
  let content = notebookTemplate
    .replace('__POLYGON__', polygonJson)
    .replace('__ZARR__', activeCollection.value?.assets.data.href ?? 'ERROR')
  let file = new File([content], 'sliced_dataset_workbench.ipynb', {
    type: 'application/x-ipynb+json',
  })

  let url = URL.createObjectURL(file)
  let a = document.createElement('a')
  a.href = url
  a.download = 'notebook.ipynb'
  a.click()

  URL.revokeObjectURL(url)
}

let pdfLink = ref('')
</script>

<template>
  <div
    style="
      position: fixed;
      width: 300px;
      left: 0;
      z-index: 100;
      background-color: white;
    "
  >
    <v-radio-group v-model="activeCollectionId">
      <v-expansion-panels :model-value="activeCollectionId">
        <v-expansion-panel
          :value="collection?.id"
          v-for="collection in collections"
          :key="collection?.id"
        >
          <v-expansion-panel-title readonly hide-actions>
            <v-radio :label="collection?.title" :value="collection?.id" />
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <div
              v-for="(summary, key) in collection?.summaries"
              style="display: flex; gap: 8px; flex-wrap: wrap"
            >
              <v-select
                :label="summary.label"
                :items="summary.options"
                item-title="label"
                item-value="value"
                density="compact"
                return-object
                v-model="variables[key]"
              />
            </div>
          </v-expansion-panel-text>
        </v-expansion-panel>
      </v-expansion-panels>
    </v-radio-group>
  </div>
  <MapboxMap
    :access-token="mapboxToken"
    map-style="mapbox://styles/anoet/cljpm695q004t01qo5s7fhf7d"
    style="height: 100vh"
    @mb-created="instantiateDraw"
  >
    <MapboxLayer
      v-if="geojson.id"
      :key="geojson.id"
      :id="geojson.id"
      :options="geojson"
    />
  </MapboxMap>

  <!--<div
    v-if="option"
    style="
      position: fixed;
      right: 0;
      bottom: 0;
      z-index: 100;
      background-color: white;
      width: 400px;
      height: 400px;
    "
  >
    <client-only>
      <v-chart :option="option" class="chart" :init-options="chartInit" />
    </client-only>
  </div>-->

  <div
    v-if="polygons?.length > 0"
    style="
      position: absolute;
      right: 32px;
      bottom: 80px;
      display: flex;
      gap: 12px;
      flex-direction: column;
    "
  >
    <v-btn @click="downloadNotebook" prepend-icon="mdi-language-python"
      >Download Notebook</v-btn
    >
    <v-btn :href="pdfLink" target="_blank" prepend-icon="mdi-file-pdf-box"
      >Download PDF</v-btn
    >
  </div>
</template>
