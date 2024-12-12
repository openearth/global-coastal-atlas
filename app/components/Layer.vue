<script setup lang="ts">
import { LayerLink } from '~/types'

let { link } = defineProps<{
  link: LayerLink
}>()
</script>

<template>
  <ItemLayer v-if="link.type === 'item'" :href="link.href" />
  <RasterLayer
    v-if="link.type === 'raster'"
    :href="link.href"
    :id="link.href"
  />
</template>
