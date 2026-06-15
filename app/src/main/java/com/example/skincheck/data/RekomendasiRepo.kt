package com.example.skincheck.data

import android.content.Context
import com.example.skincheck.helper.DatabaseHelper

class RekomendasiRepo (context: Context) {
    private val dbHelper = DatabaseHelper(context)

    fun getRecommendedIngredients(skinProblem: String): String {
        return dbHelper.getRecommendations(skinProblem) ?: "Tidak ada rekomendasi tersedia"
    }
}