import collectionShape from '../STAC/data/current/sub_threat/collection.json'
import itemShape from '../../STAC/data/current/sub_threat/eapa-mapbox/eapa-mapbox-time-2010.json'

export type CollectionType = typeof collectionShape
export type ItemType = typeof itemShape

export interface ItemLink {
  type: 'item'
  href: string
}

export interface RasterLink {
  type: 'raster'
  href: string
}

export type LayerLink = ItemLink | RasterLink
