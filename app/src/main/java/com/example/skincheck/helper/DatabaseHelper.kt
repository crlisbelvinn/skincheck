package com.example.skincheck.helper

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

class DatabaseHelper(context: Context) :
    SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_NAME = "SkinCareDB"
        private const val DATABASE_VERSION = 1
        private const val TABLE_PROBLEM = "SkincareRecommendations"
        private const val COLUMN_ID = "id"
        private const val COLUMN_PROBLEM = "skin_problem"
        private const val COLUMN_INGREDIENTS = "recommended_ingredients"

        private const val TABLE_INGREDIENT = "Ingredients"
        private const val COLUMN_INGREDIENT_NAME = "name"
        private const val COLUMN_INGREDIENT_DESC = "description"
    }

    override fun onCreate(db: SQLiteDatabase) {
        // Buat tabel untuk masalah kulit dan rekomendasi
        val createTableProblem = "CREATE TABLE $TABLE_PROBLEM (" +
                "$COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "$COLUMN_PROBLEM TEXT, " +
                "$COLUMN_INGREDIENTS TEXT)"
        db.execSQL(createTableProblem)

        // Buat tabel untuk definisi kandungan
        val createTableIngredient = "CREATE TABLE $TABLE_INGREDIENT (" +
                "$COLUMN_ID INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "$COLUMN_INGREDIENT_NAME TEXT, " +
                "$COLUMN_INGREDIENT_DESC TEXT)"
        db.execSQL(createTableIngredient)

        // Insert data awal
        insertInitialData(db)
        insertIngredientsData(db)
    }

    private fun insertInitialData(db: SQLiteDatabase) {
        val problems = listOf(
            "INSERT INTO $TABLE_PROBLEM ($COLUMN_PROBLEM, $COLUMN_INGREDIENTS) VALUES ('Bekas Jerawat', 'Niacinamide, Vitamin C, Alpha Arbutin')",
            "INSERT INTO $TABLE_PROBLEM ($COLUMN_PROBLEM, $COLUMN_INGREDIENTS) VALUES ('Hiperpigmentasi', 'Kojic Acid, Vitamin C, Licorice Extract')",
            "INSERT INTO $TABLE_PROBLEM ($COLUMN_PROBLEM, $COLUMN_INGREDIENTS) VALUES ('Jerawat', 'Salicylic Acid, Benzoyl Peroxide, Tea Tree Oil')",
            "INSERT INTO $TABLE_PROBLEM ($COLUMN_PROBLEM, $COLUMN_INGREDIENTS) VALUES ('Kerutan', 'Retinol, Peptides, Collagen')",
            "INSERT INTO $TABLE_PROBLEM ($COLUMN_PROBLEM, $COLUMN_INGREDIENTS) VALUES ('Komedo', 'Salicylic Acid, Clay Mask, Niacinamide')"
        )
        for (query in problems) {
            db.execSQL(query)
        }
    }

    private fun insertIngredientsData(db: SQLiteDatabase) {
        val ingredients = listOf(
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Niacinamide', 'Vitamin B3 untuk mencerahkan kulit dan memperkuat skin barrier')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Vitamin C', 'Antioksidan yang mencerahkan dan merangsang produksi kolagen')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Alpha Arbutin', 'Mencerahkan kulit dan menyamarkan noda hitam')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Kojic Acid', 'Mencerahkan kulit dan mengurangi hiperpigmentasi')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Licorice Extract', 'Mencerahkan kulit dan menenangkan iritasi')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Salicylic Acid', 'Membersihkan pori-pori dan mengatasi jerawat')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Benzoyl Peroxide', 'Membunuh bakteri penyebab jerawat')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Tea Tree Oil', 'Antimikroba alami untuk meredakan jerawat')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Retinol', 'Meningkatkan regenerasi sel dan mengurangi kerutan')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Peptides', 'Merangsang produksi kolagen untuk kulit kencang')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Collagen', 'Menjaga elastisitas dan kelembapan kulit')",
            "INSERT INTO $TABLE_INGREDIENT ($COLUMN_INGREDIENT_NAME, $COLUMN_INGREDIENT_DESC) VALUES ('Clay Mask', 'Menyerap minyak dan membersihkan pori-pori')"
        )
        for (query in ingredients) {
            db.execSQL(query)
        }
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS $TABLE_PROBLEM")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_INGREDIENT")
        onCreate(db)
    }

    // Fungsi untuk mendapatkan rekomendasi kandungan berdasarkan masalah kulit
    fun getRecommendations(skinProblem: String): String? {
        val db = readableDatabase
        val query = "SELECT $COLUMN_INGREDIENTS FROM $TABLE_PROBLEM WHERE $COLUMN_PROBLEM = ?"
        val cursor = db.rawQuery(query, arrayOf(skinProblem))

        var result: String? = null
        if (cursor.moveToFirst()) {
            result = cursor.getString(0)
        }
        cursor.close()
        return result
    }

    // Fungsi untuk mendapatkan deskripsi berdasarkan nama kandungan
    fun getIngredientDescription(ingredient: String): String? {
        val db = readableDatabase
        val query = "SELECT $COLUMN_INGREDIENT_DESC FROM $TABLE_INGREDIENT WHERE $COLUMN_INGREDIENT_NAME = ?"
        val cursor = db.rawQuery(query, arrayOf(ingredient))

        var result: String? = null
        if (cursor.moveToFirst()) {
            result = cursor.getString(0)
        }
        cursor.close()
        return result
    }
}