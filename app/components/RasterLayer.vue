<script setup lang="ts">
import { MapboxLayer } from '@studiometa/vue-mapbox-gl'

let { id, href } = defineProps<{
  id: string
  href: string
}>()

let layer = computed(() => {
  return {
    id,
    type: 'raster',
    source: {
      type: 'raster',
      tiles: [href],
      tileSize: 256,
    },
  }
})
</script>

<template>
  <MapboxLayer v-if="layer" :key="layer.id" :id="layer.id" :options="layer" />
</template>
