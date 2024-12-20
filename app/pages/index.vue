<script setup lang="ts">
import 'mapbox-gl/dist/mapbox-gl.css'
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css'
import notebookTemplate from '~/assets/sliced_dataset_workbench.ipynb?raw'
import * as turf from '@turf/turf'

import collectionShape from '../../STAC/data/current/sub_threat/collection.json'

import CollectionSelector from '~/components/CollectionSelector.vue'
import { LayerLink } from '~/types'
import Layer from '~/components/Layer.vue'
import { BookOpen, Hammer, Info, Loader, Route, X } from 'lucide-vue-next'

import Map from '~/components/Map.vue'
import { useCollections } from '~/composables/useCollections'

type CollectionType = typeof collectionShape

let {
  public: { mapboxToken, pdfEndpoint },
} = useRuntimeConfig()

let url = useRequestURL()

let headers = useRequestHeaders()

let collectionLinks = [
  'https://raw.githubusercontent.com/openearth/coclicodata/8aabe3516bdb287d9972618d28e6471b7a69adf9/current/cfhp/collection.json',
  'https://raw.githubusercontent.com/openearth/coclicodata/8aabe3516bdb287d9972618d28e6471b7a69adf9/current/slp/collection.json',
  'https://raw.githubusercontent.com/openearth/coclicodata/62ccb63944edaaadecb140eca57003a3b95d091d/current/deltares-delta-dtm/collection.json',
  url.protocol + '//' + url.host + '/collections/subsidence.json',
]

let collections = await useCollections({ collectionLinks })

console.log(collections.value, collectionLinks)

let polygons = ref([])
let selectedCollections = ref<string[]>([])

const mapComponent = ref<InstanceType<typeof Map>>()

async function downloadNotebook() {
  if (!process.client) return

  let polygonJson = JSON.stringify(
    mapComponent.value?.draw?.getAll().features[0].geometry,
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

  if (polygons.value.length > 0) {
    let polygonJson = encodeURIComponent(
      JSON.stringify(polygons.value[0].geometry),
    )
    let pdfLink = `${pdfEndpoint}?polygon=${polygonJson}`

    isLoadingPdf.value = true
    let file = await $fetch(pdfLink, {
      headers,
    })

    let url = URL.createObjectURL(file)
    window.open(url)
    isLoadingPdf.value = false
  }
}

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

let { data: countries } = await useFetch(
  url.protocol + '//' + url.host + '/countries.json',
)

// Find countries that intersect with drawn polygons
let intersectingCountries = computed(() => {
  if (!polygons.value?.length) return []

  const drawnPolygon = polygons.value[0]

  return countries.value?.features.filter((country) => {
    return turf.intersect(drawnPolygon, country.geometry) !== null
  })
})

function setDrawMode(mode: string) {
  mapComponent.value?.draw?.changeMode(mode)
}

function trashPolygons() {
  mapComponent.value?.draw?.deleteAll()
  polygons.value = []
}

function formatArea(polygons: any) {
  let area = turf.area(polygons[0])
  return area >= 100_000_000_000
    ? (area / 1000_000_000_000).toFixed(2) + ' million km²'
    : area / 1000_000 > 1000
    ? (area / 1000_000).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, '.')
    : (area / 1000_000).toFixed(0) + ' km²'
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

      <v-bottom-sheet>
        <template v-slot:activator="{ props }">
          <button
            v-bind="props"
            class="flex items-center justify-center h-full gap-1.5 text-gray-600 hover:text-gray-900 focus:text-gray-900"
          >
            <Info class="size-4" /> About
          </button>
        </template>

        <v-card
          title="About this project"
          text="Lorem ipsum dolor sit amet consectetur, adipisicing elit. Ut, eos? Nulla aspernatur odio rem, culpa voluptatibus eius debitis dolorem perspiciatis asperiores sed consectetur praesentium! Delectus et iure maxime eaque exercitationem!"
        ></v-card>
      </v-bottom-sheet>
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
    <Map
      ref="mapComponent"
      :mapbox-token="mapboxToken"
      map-style="mapbox://styles/anoet/cljpm695q004t01qo5s7fhf7d"
      v-model:polygons="polygons"
    >
      <template
        v-for="itemLink in Object.values(itemLinks)"
        :key="itemLink?.href"
      >
        <Layer v-if="itemLink" :link="itemLink" />
      </template>
    </Map>
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
        @click="setDrawMode('draw_polygon')"
        class="mt-3"
      >
        <!-- <template v-if="drawMode === 'draw_polygon'">
          <Loader class="size-4 animate-spin mr-1.5" /> Waiting for
          drawing&hellip;
        </template> -->
        <Route class="size-4 mr-1.5" /> Draw Polygon
      </v-btn>
    </div>

    <template v-if="polygons?.length">
      <div class="mb-4 p-3 bg-gray-50 rounded-lg">
        <div class="text-sm text-gray-600">
          Selected Area:
          {{ formatArea(polygons) }}

          <div
            v-if="intersectingCountries?.length"
            class="font-medium text-black"
          >
            {{
              intersectingCountries
                .slice(0, 10)
                .map((country) => country.properties.ADMIN)
                .join(', ')
            }}

            <span v-if="intersectingCountries.length > 10">
              and {{ intersectingCountries.length - 10 }} more
            </span>
          </div>
        </div>
      </div>

      <div class="flex gap-3 mb-4">
        <v-btn
          color="secondary"
          :loading="isLoadingPdf"
          @click="downloadPdf"
          prepend-icon="mdi-file-pdf-box"
          class="flex-1"
        >
          Download Report
        </v-btn>

        <v-btn variant="outlined" @click="trashPolygons">
          <X class="size-4" />
        </v-btn>
      </div>

      <hr class="my-6" />

      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold">Available Datasets</h3>
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
      </div>
    </template>
  </div>
</template>
