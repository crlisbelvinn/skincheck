package com.example.skincheck.view


import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import okhttp3.OkHttpClient

//object ApiClient {
//    private const val BASE_URL = "https://api-inference.huggingface.co/"
//
//    private val client = OkHttpClient.Builder().build()
//
//    private val retrofit: Retrofit = Retrofit.Builder()
//        .baseUrl(BASE_URL)
//        .client(client)
//        .addConverterFactory(GsonConverterFactory.create())
//        .build()
//
//    val huggingFaceApi: HuggingFaceApi = retrofit.create(HuggingFaceApi::class.java)
//}

object ApiClient {
    private const val BASE_URL = "https://belpin-belpin-vitmodel-skincheck.hf.space/"

    private val client = OkHttpClient.Builder().build()

    private val retrofit: Retrofit = Retrofit.Builder()
        .baseUrl(BASE_URL)
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()

    val huggingFaceApi: HuggingFaceApi = retrofit.create(HuggingFaceApi::class.java)
}
