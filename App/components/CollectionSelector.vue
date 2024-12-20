<script lang="ts" setup>
import { Eye, EyeOff, Info } from 'lucide-vue-next'
import { CollectionType, LayerLink } from '../types'
import type { Ref } from 'vue'

let { collection, onChangeActive, activeValue } = defineProps<{
  collection: CollectionType & { href: string }
  activeValue?: LayerLink
  onChangeActive(link?: LayerLink): void
}>()

let baseURL = collection.href.replace('/collection.json', '')

let summaries = computed(() => {
  return collection?.summaries
})

let propertyOrder = computed(() => Object.keys(summaries.value))

let variables: Ref<Record<string, string | undefined>> = ref({})

function resetVariables() {
  variables.value = Object.keys(summaries.value ?? {}).reduce((acc, key) => {
    return {
      ...acc,
      [key]: undefined,
    }
  }, {} as Record<string, string | undefined>)
}

resetVariables()

function getValidOptionsForProperty(property: string) {
  if (!variables.value) return []

  let currentIndex = propertyOrder.value.indexOf(property)
  let relevantProperties = propertyOrder.value.slice(0, currentIndex)

  let validItems = collection.links
    .filter((link) => link.rel === 'item')
    .filter((link) => {
      if (!variables.value) return false
      return Object.entries(variables.value)
        .filter(([key]) => relevantProperties.includes(key))
        .every(
          ([key, option]) =>
            !option ||
            link.properties?.[key as keyof typeof link.properties] === option,
        )
    })

  return [
    ...new Set(
      validItems.map(
        (item) => item?.properties?.[property as keyof typeof item.properties],
      ),
    ),
  ]
}

function selectOption(property: string, option: string) {
  if (!variables.value) return

  if (variables.value[property] === option) {
    variables.value[property] = undefined
  } else {
    variables.value[property] = option
  }

  let currentIndex = propertyOrder.value.indexOf(property)
  for (let i = currentIndex + 1; i < propertyOrder.value.length; i++) {
    let nextProperty = propertyOrder.value[i]
    if (!variables.value) break

    variables.value[nextProperty] = undefined

    let validOptions = getValidOptionsForProperty(nextProperty)
    if (validOptions.length === 1) {
      variables.value[nextProperty] = validOptions[0]
    }
  }
}

watchEffect(() => {
  if (!summaries.value || !variables.value) return

  let foundLink = collection.links
    .filter((l) => l.rel === 'item')
    .find((link) => {
      return Object.entries(variables.value ?? {}).every(
        ([key, option]) =>
          link.properties?.[key as keyof typeof link.properties] === option,
      )
    })

  if (foundLink) {
    onChangeActive({
      type: 'item',
      href: baseURL + '/' + foundLink.href,
    })
  } else {
    onChangeActive(undefined)
  }
})

function toggleActive(value: boolean) {
  if (!value) {
    onChangeActive(undefined)
    resetVariables()
  } else {
    if (summaries.value) {
      // do nothing
    } else {
      const geoserverLink = collection.assets?.['geoserver_link'] as
        | { href: string }
        | undefined
      if (geoserverLink) {
        onChangeActive({
          type: 'raster',
          href: geoserverLink.href,
        })
      }
    }
  }
}
</script>

<template>
  <div v-if="!summaries" class="w-full">
    <div
      class="flex items-center gap-3 bg-white p-3 shadow w-full justify-between px-6"
    >
      <div class="text-sm font-medium">{{ collection.title }}</div>
      <div class="flex items-center gap-1.5">
        <v-tooltip :text="collection.description" class="max-w-[640px]">
          <template v-slot:activator="{ props }">
            <Info v-bind="props" class="size-3.5 shrink-0" />
          </template>
        </v-tooltip>
        <button
          @click="toggleActive(!activeValue)"
          class="size-8 flex items-center justify-center shrink-0 rounded-md hover:bg-gray-100"
        >
          <Eye class="size-4" v-if="!!activeValue" />
          <EyeOff class="size-4" v-if="!activeValue" />
        </button>
      </div>
    </div>
  </div>
  <v-expansion-panel v-if="summaries">
    <v-expansion-panel-title>
      <div class="flex items-center justify-between w-full gap-3">
        <div>{{ collection.title }}</div>
        <div class="flex items-center gap-1.5 z-10">
          <v-tooltip :text="collection.description" class="max-w-[640px]">
            <template v-slot:activator="{ props }">
              <Info v-bind="props" class="size-3.5 shrink-0" />
            </template>
          </v-tooltip>
          <button
            @click.stop="!!activeValue && toggleActive(false)"
            class="z-10 size-8 flex items-center justify-center shrink-0 rounded-md hover:bg-gray-100"
          >
            <Eye class="size-4" v-if="!!activeValue" />
            <EyeOff class="size-4" v-if="!activeValue" />
          </button>
        </div>
      </div>
    </v-expansion-panel-title>
    <v-expansion-panel-text>
      <div class="grid grid-flow-row gap-1.5 py-3">
        <div
          v-for="(options, key) in summaries"
          :key="key"
          class="empty:hidden"
        >
          <v-select
            :label="key"
            :items="getValidOptionsForProperty(key)"
            :model-value="variables?.[key]"
            @update:model-value="selectOption(key, $event)"
          />
        </div>
      </div>
    </v-expansion-panel-text>
  </v-expansion-panel>
</template>
