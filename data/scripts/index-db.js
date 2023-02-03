db.reviews.createIndex(
    {
        overall: 1,
    },
    {
        name: "rating_index"
    }
)
