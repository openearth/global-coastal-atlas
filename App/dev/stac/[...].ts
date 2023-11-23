import fs from 'node:fs/promises'

export default defineEventHandler(async (event) => {
  let path = event.context.params?.['_']

  if (!path) throw new Error('No path provided.')

  let json = await fs.readFile(`${process.cwd()}/../STAC/data/current/${path}`)

  return json
})
