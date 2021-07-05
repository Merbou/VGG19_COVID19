<template>
  <div class="flex-container">
    <el-row :gutter="12">
      <el-col :span="14">
        <el-card shadow="hover">
          <div slot="header" class="clearfix">
            <h3 class="res_content">Browser to the CXR image</h3>
            <el-button
              type="primary"
              round
              size="mini"
              plain
              @click="validateBefore"
              >Analysis</el-button
            >
          </div>
          <el-form :rules="rules" ref="ruleForm" :model="file">
            <el-form-item prop="imgs">
              <el-upload
                action="#"
                list-type="picture-card"
                :on-preview="handlePictureCardPreview"
                :on-remove="handleRemove"
                :on-change="handleChange"
                :auto-upload="false"
                :file-list="file.imgs"
                :limit="100"
                multiple
              >
                <i class="el-icon-plus"></i>
              </el-upload>
            </el-form-item>
          </el-form>

          <el-dialog :visible.sync="dialogVisible">
            <img width="100%" :src="dialogImageUrl" alt="" />
          </el-dialog>
        </el-card>
      </el-col>
    </el-row>
    <el-row :gutter="12" v-if="!!predicted_list && predicted_list.length > 0">
      <el-col :span="22">
        <el-card shadow="always">
          <div slot="header" class="clearfix">
            <h3>Analysis result</h3>
            <Export
              :files="file.imgs"
              :labels="labels"
              :predectedList="predicted_list"
            />
          </div>

          <Echarts
            :files="file.imgs"
            :labels="labels"
            :predectedList="predicted_list"
          />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { post } from "axios";
import Echarts from "./Echarts.vue";
import Export from "./Export.vue";
export default {
  components: {
    Echarts,
    Export,
  },
  data() {
    let validateImage = (rule, value, callback) => {
      rule;
      // const regex = new RegExp("(^image)/");
      // value.forEach((v) => {
      //   if (v && v.raw && !regex.test(v.raw.type))
      //     callback(new Error("CXR Image must be JPG format!"));
      // });
      callback();
    };

    return {
      labels: [],
      predicted_list: [],
      dialogImageUrl: "",
      dialogVisible: false,
      file: { imgs: [] },
      fullscreenLoading: false,
      rules: {
        imgs: [
          {
            validator: validateImage,
            trigger: "change",
          },
        ],
      },
    };
  },
  computed: {
    result() {
      if (this.index) return this.labels[this.index];
      return "";
    },
  },
  methods: {
    validateBefore() {
      this.$refs["ruleForm"].validate((valid) => {
        if (!valid) {
          return false;
        } else this.dispatch();
      });
    },
    dispatch() {
      let formdata = new FormData();

      const files = this.file.imgs;
      if (files.length <= 0) return;

      const loading = this.$loading({
        lock: true,
        text: "Loading",
        spinner: "el-icon-loading",
        background: "rgba(0, 0, 0, 0.7)",
      });

      Array.from(files).forEach((file, index) => {
        formdata.append(`img_${index}`, file.raw);
      });
      post("http://127.0.0.1:5000/predict", formdata, {
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then(({ data }) => {
          let { labels, predicted_list } = data;
          this.labels = labels;
          this.predicted_list = predicted_list;
        })
        .catch((err) => {
          let data = err.response && err.response.data;
          data && this.$message.error(data.errors);
        })
        .finally(() => {
          loading.close();
        });
    },
    handleRemove(file) {
      const index = this.file.imgs.findIndex((e) => e.uid === file.uid);
      this.file.imgs.splice(index, 1);
    },
    handlePictureCardPreview(file) {
      this.dialogImageUrl = file.url;
      this.dialogVisible = true;
    },
    handleChange(file, fileList) {
      if (fileList && fileList.length > 100)
        this.$message.error("Do not upload more than 100 images");
      fileList = fileList.slice(0, 100);
      file;
      this.file.imgs = fileList;
    },
  },
};
</script>
<style>
.el-row {
  display: flex;
  justify-content: center;
  padding: 15px;
}
.clearfix {
  display: flex;
  justify-content: space-between;
}
.res_content {
  display: flex;
  align-items: center;
}
</style>