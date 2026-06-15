package com.example.skincheck.view

import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Response
import retrofit2.http.*

//interface HuggingFaceApi {
//    @POST("https://belpin-belpin-vitmodel-skincheck.hf.space/predict")
//    suspend fun classifyImage(
//        @Header("Authorization") authHeader: String,
//        @Body image: RequestBody
//    ): Response<List<Prediction>>
//}
interface HuggingFaceApi {
    @Multipart
    @POST("predict")
    suspend fun classifyImage(
        @Part file: MultipartBody.Part
    ): Response<Prediction>
}
