import { CollectionType } from '~/types'

export async function useCollections({
  collectionLinks,
}: {
  collectionLinks: string[]
}) {
  let headers = useRequestHeaders()

  const { data: collections } = await useAsyncData('collections', async () => {
    return Promise.all(
      collectionLinks.map(async (collectionLink) => {
        const res = await fetch(collectionLink, {
          headers: {
            ...headers,
            Accept: 'application/json',
          },
        })
        const text = await res.text()
        const data = JSON.parse(text)
        return { ...data, href: collectionLink } as CollectionType & {
          href: string
        }
      }),
    )
  })

  return collections
}
