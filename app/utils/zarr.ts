import { openArray } from 'zarr'
import * as turf from '@turf/turf'

async function getDimension(path: string) {
  let z = await openArray({
    store: 'https://storage.googleapis.com/dgds-data-public/gca/ESLbyGWL.zarr/',
    path,
    mode: 'r',
  })

  return (await z.get(':')).data as Float64Array
}

export async function getDataByPolygon(searchWithin: turf.Polygon) {
  let lats = await getDimension('lat')
  let lons = await getDimension('lon')

  let coords = [...lats].map((lat, i) => [lons[i], lat])
  let points = turf.points(coords)

  let pointsWithin = turf.pointsWithinPolygon(points, searchWithin)

  let indices = pointsWithin.features.map((f) =>
    coords.findIndex(
      (c) =>
        c[0] === f.geometry.coordinates[0] &&
        c[1] === f.geometry.coordinates[1],
    ),
  )

  // console.log(indices)

  let dataArray = await openArray({
    store: 'https://storage.googleapis.com/dgds-data-public/gca/ESLbyGWL.zarr/',
    path: 'esl',
    mode: 'r',
  })

  let allData = await dataArray.get([':', ':', 0, ':'])

  let rpValues = await getDimension('rp')
  let gwlValues = await getDimension('gwl')
  let ensembleValues = await getDimension('ensemble')

  // let groupedData: Record<string, Record<string, number[]>> = {}
  // let series = []
  // for (let rp of rpValues) {
  //   groupedData[rp] = {}

  //   for (let gwl of gwlValues) {
  //     let { data } = await dataArray.get([
  //       rpValues.indexOf(rp),
  //       gwlValues.indexOf(gwl),
  //       0,
  //     ])

  //     series.push = {
  //       name: `GWL ${gwl}`,
  //       type: 'line',
  //       data: indices.map((i) => data[i]),
  //     }
  //     groupedData[rp][gwl] = indices.map((i) => data[i])
  //   }
  // }

  return {
    rpValues,
    gwlValues,
    ensembleValues,
    dataArray,
    indices,
    allData,
    // groupedData,
  }
}
