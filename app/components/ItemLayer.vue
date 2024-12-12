<script setup lang="ts">
import { ItemType } from '~/types'

let { href } = defineProps<{
  href: string
}>()

let { data: jsonString } = await useFetch<ItemType>(href)

let item = computed(() => JSON.parse(jsonString.value))
</script>

<template>
  <RasterLayer :id="item.id" :href="item.assets.visual.href" />
</template>
