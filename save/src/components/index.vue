<template>
  <div class="flex-container">
    <el-row :gutter="12">
      <el-col :span="14">
        <el-card shadow="hover">
          <div slot="header" class="clearfix">
            <h3 class="res_content">Browser to the CXR image</h3>
            <el-button type="primary" round size="mini" plain @click="dispatch">Analysis</el-button>
          </div>
          <el-upload
            action="#"
            list-type="picture-card"
            :on-preview="handlePictureCardPreview"
            :on-remove="handleRemove"
            :auto-upload="false"
            :file-list="fileList"
            multiple
            :on-change="handleChange"
          >
            <i class="el-icon-plus"></i>
          </el-upload>
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
              :files="fileList"
              :labels="labels"
              :predectedList="predicted_list"
            />
          </div>

          <Echarts
            :files="fileList"
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
    return {
      labels: [],
      predicted_list: [],
      dialogImageUrl: "",
      dialogVisible: false,
      fileList: [],
      fullscreenLoading: false,
    };
  },
  computed: {
    result() {
      if (this.index) return this.labels[this.index];
      return "";
    },
  },
  methods: {
    dispatch() {
      let formdata = new FormData();

      const files = this.fileList;
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
        .then(() => {})
        .finally(() => {
          loading.close();
        });
    },
    handleRemove(file) {
      const index = this.fileList.findIndex((e) => e.uid === file.uid);
      this.fileList.splice(index, 1);
    },
    handlePictureCardPreview(file) {
      this.dialogImageUrl = file.url;
      this.dialogVisible = true;
    },
    handleChange(file, fileList) {
      file;
      this.fileList = fileList;
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