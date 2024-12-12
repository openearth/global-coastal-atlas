<script setup lang="ts">
import { MapboxMap } from '@studiometa/vue-mapbox-gl'
import MapboxDraw from '@mapbox/mapbox-gl-draw'
import 'mapbox-gl/dist/mapbox-gl.css'
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css'
import notebookTemplate from '~/assets/sliced_dataset_workbench.ipynb?raw'
import * as turf from '@turf/turf'

import collectionShape from '../../STAC/data/current/sub_threat/collection.json'

import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
} from 'echarts/components'
import { THEME_KEY } from 'vue-echarts'
import CollectionSelector from '~/components/CollectionSelector.vue'
import { LayerLink } from '~/types'
import Layer from '~/components/Layer.vue'
import {
  BookOpen,
  Hammer,
  Home,
  Info,
  Loader,
  Route,
  Trash,
} from 'lucide-vue-next'

type CollectionType = typeof collectionShape

let {
  public: { mapboxToken, pdfEndpoint },
} = useRuntimeConfig()

// let baseURL = url.protocol + '//' + url.host + '/stac'

// let catalogPath = `${baseURL}/catalog.json`

let headers = useRequestHeaders()

// let { data: catalogJson } = await useFetch<CatalogType>(catalogPath, {
//   headers,
// })

// let catalog = catalogJson.value

// let childrenLinks = catalog?.links.filter((link) => link.rel === 'child') ?? []

let collectionLinks = [
  'https://raw.githubusercontent.com/openearth/coclicodata/8aabe3516bdb287d9972618d28e6471b7a69adf9/current/cfhp/collection.json',
  'https://raw.githubusercontent.com/openearth/coclicodata/8aabe3516bdb287d9972618d28e6471b7a69adf9/current/slp/collection.json',
  'https://raw.githubusercontent.com/openearth/coclicodata/62ccb63944edaaadecb140eca57003a3b95d091d/current/deltares-delta-dtm/collection.json',
]

const { data: collections } = await useAsyncData('collections', async () => {
  return Promise.all(
    collectionLinks.map(async (collectionLink) => {
      const res = await fetch(collectionLink, {
        headers: {
          ...headers,
          Accept: 'application/json',
        },
      })
      const text = await res.text()
      const data = JSON.parse(text)
      return { ...data, href: collectionLink } as CollectionType & {
        href: string
      }
    }),
  )
})

let draw = ref<MapboxDraw | null>(null)
let polygons = ref([])
let selectedCollections = ref<string[]>([])

function instantiateDraw(map) {
  if (!process.client) return

  draw.value = new MapboxDraw({
    displayControlsDefault: false,
    defaultMode: 'draw_polygon',
  })

  map.addControl(draw.value)

  map.on('draw.create', updateArea)
  map.on('draw.delete', updateArea)
  map.on('draw.update', updateArea)

  async function updateArea(e) {
    const data = draw.value.getAll()
    polygons.value = data.features

    if (data.features.length > 0) {
      let polygonJson = encodeURIComponent(
        JSON.stringify(data.features[0].geometry),
      )
      pdfLink.value = `${pdfEndpoint}?polygon=${polygonJson}`
    } else {
      pdfLink.value = ''
      selectedCollections.value = []
    }
  }
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
  if (!process.client) return

  let polygonJson = JSON.stringify(
    draw.value?.getAll().features[0].geometry,
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

let isLoadingPdf = ref(false)
async function downloadPdf() {
  if (!process.client) return

  isLoadingPdf.value = true
  let file = await $fetch(`${pdfLink.value}`, {
    headers,
  })

  let url = URL.createObjectURL(file)
  // let a = document.createElement('a')
  // a.href = url
  // a.download = 'report.pdf'
  // a.click()

  // URL.revokeObjectURL(url)
  let w = window.open(url)
  isLoadingPdf.value = false
}

let pdfLink = ref('')

let itemLinks = ref<Record<string, LayerLink>>({})

function isCollectionIntersecting(collection: CollectionType) {
  if (!polygons.value?.length) return false
  if (!collection.extent?.spatial?.bbox?.[0]) return false

  const bbox = turf.bbox(polygons.value[0])
  const collectionBboxPolygon = turf.bboxPolygon(
    collection.extent.spatial.bbox[0],
  )
  const drawnBboxPolygon = turf.bboxPolygon(bbox)

  return turf.intersect(collectionBboxPolygon, drawnBboxPolygon) !== null
}
</script>

<template>
  <div
    class="p-3 fixed top-0 left-0 w-full z-10 flex items-center justify-center"
  >
    <div
      class="bg-white shadow-lg rounded h-10 px-5 text-sm flex items-center justify-center gap-8"
    >
      <a
        href="https://github.com/Deltares-research/IDP-workbench"
        target="_blank"
        rel="noopener noreferrer"
        class="flex items-center justify-center h-full gap-1.5 text-gray-600 hover:text-gray-900 focus:text-gray-900"
      >
        <Hammer class="size-4" /> Workbench
      </a>

      <a
        href="https://publicwiki.deltares.nl/spaces/viewspace.action?key=LTSV"
        target="_blank"
        rel="noopener noreferrer"
        class="flex items-center justify-center h-full gap-1.5 text-gray-600 hover:text-gray-900 focus:text-gray-900"
      >
        <BookOpen class="size-4" /> Wiki
      </a>

      <a
        href="https://github.com/Deltares-research/IDP-workbench"
        target="_blank"
        rel="noopener noreferrer"
        class="flex items-center justify-center h-full gap-1.5 text-gray-600 hover:text-gray-900 focus:text-gray-900"
      >
        <Info class="size-4" /> About
      </a>
    </div>
  </div>

  <div
    class="h-fit max-h-full overflow-y-auto fixed w-[320px] top-3 left-3 z-10 rounded"
  >
    <client-only>
      <v-expansion-panels class="rounded shadow-lg">
        <CollectionSelector
          v-for="collection in collections || []"
          :key="collection.id"
          :collection="collection"
          :active-value="itemLinks[collection.id]"
          @change-active="itemLinks[collection.id] = $event"
        />
      </v-expansion-panels>
    </client-only>
  </div>
  <client-only>
    <MapboxMap
      :access-token="mapboxToken"
      map-style="mapbox://styles/anoet/cljpm695q004t01qo5s7fhf7d"
      style="height: 100vh"
      @mb-created="instantiateDraw"
    >
      <template
        v-for="itemLink in Object.values(itemLinks)"
        :key="itemLink?.href"
      >
        <Layer v-if="itemLink" :link="itemLink" />
      </template>
    </MapboxMap>
  </client-only>

  <div
    class="fixed right-3 top-3 w-[320px] bg-white shadow-lg p-4 z-10 overflow-y-auto rounded"
  >
    <div v-if="!polygons?.length" class="text-center p-4">
      <p class="text-sm text-gray-600">
        Draw a polygon on the map to select your area of interest
      </p>
      <v-btn
        variant="outlined"
        @click="draw?.changeMode('draw_polygon')"
        class="mt-3"
      >
        <template v-if="draw?.getMode() === 'draw_polygon'">
          <Loader class="size-4 animate-spin mr-1.5" /> Waiting for
          drawing&hellip;
        </template>
        <template v-else>
          <Route class="size-4 mr-1.5" /> Draw Polygon
        </template>
      </v-btn>
    </div>

    <template v-if="polygons?.length">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold">Available Datasets</h3>
        <v-btn variant="outlined" @click="draw?.trash()">
          <Trash class="size-4 mr-1.5" /> Clear
        </v-btn>
      </div>

      <div v-for="collection in collections" :key="collection.id">
        <v-checkbox
          v-model="selectedCollections"
          :value="collection.id"
          :label="collection.title || collection.id"
          :disabled="!isCollectionIntersecting(collection)"
          :hint="
            !isCollectionIntersecting(collection)
              ? 'Dataset not available in selected area'
              : undefined
          "
          :persistent-hint="!isCollectionIntersecting(collection)"
        />
      </div>

      <div class="mt-6 flex flex-col gap-3">
        <v-btn
          color="primary"
          block
          disabled
          @click="downloadNotebook"
          prepend-icon="mdi-language-python"
          title="Coming soon"
        >
          Analyze in Notebook
        </v-btn>

        <v-btn
          color="secondary"
          block
          :disabled="!selectedCollections.length"
          :loading="isLoadingPdf"
          @click="downloadPdf"
          prepend-icon="mdi-file-pdf-box"
        >
          Download Report
        </v-btn>
      </div>
    </template>
  </div>
</template>
