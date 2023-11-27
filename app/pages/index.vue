<script setup lang="ts">
import { MapboxMap, MapboxLayer } from '@studiometa/vue-mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

import itemShape from '../../STAC/data/current/sub_threat/epsi-mapbox/epsi-mapbox-time-2010.json'
import catalogShape from '../../STAC/data/current/catalog.json'
import collectionShape from '../../STAC/data/current/sub_threat/collection.json'

type ItemType = typeof itemShape
type CatalogType = typeof catalogShape
type CollectionType = typeof collectionShape

let url = useRequestURL()

let {
  public: { mapboxToken },
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
  return Object.entries(
    activeCollection.value?.['cube:variables'] || {},
  ).filter(([key, value]) => {
    return value.type !== 'data'
  })
})

// let { data: activeCollection } = await useFetch(currentCollectionPath);

// let activeCollection = ref(currentCollection);

let variables = ref(
  Object.entries(summaries.value).reduce((acc, [key, values]) => {
    return {
      ...acc,
      [key]: values[0],
    }
  }, {} as Record<string, string>),
)

watchEffect(
  () => {
    variables.value = Object.entries(summaries.value).reduce(
      (acc, [key, values]) => {
        return {
          ...acc,
          [key]: values[0],
        }
      },
      {} as Record<string, string>,
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
          ([key, value]) => link.properties?.[key] === value,
        )
      }) ?? activeCollection.value.links.filter((l) => l.rel === 'item')[0]

  return foundLink?.href
})

let { data } = await useAsyncData(
  () => $fetch(`${baseURL}/${activeCollectionId.value}/${activeItemUrl.value}`),
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
          :value="collection.id"
          v-for="collection in collections"
          :key="collection.id"
        >
          <v-expansion-panel-title readonly hide-actions>
            <v-radio :label="collection.title" :value="collection.id" />
          </v-expansion-panel-title>
          <v-expansion-panel-text>
            <div
              v-for="(options, key) in collection.summaries"
              style="display: flex; gap: 8px; flex-wrap: wrap"
            >
              <v-select
                :label="key"
                :items="options"
                density="compact"
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
  >
    <MapboxLayer
      v-if="geojson.id"
      :key="geojson.id"
      :id="geojson.id"
      :options="geojson"
    />
  </MapboxMap>
</template>
