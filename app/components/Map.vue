<script setup lang="ts">
import { MapboxMap } from '@studiometa/vue-mapbox-gl'
import MapboxDraw from '@mapbox/mapbox-gl-draw'
import 'mapbox-gl/dist/mapbox-gl.css'
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css'

defineProps<{
  mapboxToken: string
  mapStyle: string
}>()

const emit = defineEmits<{
  'update:polygons': [polygons: any[]]
}>()

let draw = ref<MapboxDraw | null>(null)

function instantiateDraw(map: any) {
  if (!process.client) return

  draw.value = new MapboxDraw({
    displayControlsDefault: false,
  })

  map.addControl(draw.value)

  map.on('draw.create', updateArea)
  map.on('draw.delete', updateArea)
  map.on('draw.update', updateArea)
}

function updateArea() {
  const data = draw.value?.getAll()
  emit('update:polygons', data?.features || [])
}

defineExpose({ draw })
</script>

<template>
  <client-only>
    <MapboxMap
      :access-token="mapboxToken"
      :map-style="mapStyle"
      style="height: 100vh"
      @mb-created="instantiateDraw"
    >
      <slot />
    </MapboxMap>
  </client-only>
</template>
